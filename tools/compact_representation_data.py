#!/usr/bin/env python3
"""Prepare fresh pilot positions and validate complete-turn search labels.

This bridge never starts games or labeling. It accepts only the new training
trajectory schema, then emits a source-bound command for the existing native
teacher. Historical campaign corpora and live replay schemas are rejected.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import gzip
import resource
import sys
import time

import compact_representation as representation
import jacek_replay_corpus as corpus
import jacek_replay_features as features
import top_three_experiments as experiments

SCHEMA = "papersoccer.compact-representation.fresh-data.v1"
CAMPAIGN_ID = "top-three-representation-20260916"
TEACHER_SHA256 = "f7bdb201a377c04531f1ba98fd73457f7f77961aa0f0f9b1ac32c59b6e85ee75"
HEADER = "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix\n"


def record(path):
    path = Path(path).resolve()
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024): digest.update(block); size += len(block)
    return {"path": str(path), "bytes": size, "sha256": digest.hexdigest()}


def bound(item):
    path = Path(item["path"])
    if record(path) != item: raise ValueError("fresh input identity changed")
    return path


def once(path, payload):
    if path.exists():
        if path.read_bytes() != payload: raise ValueError("fresh preparation already exists with another identity")
    else: path.write_bytes(payload)


def feature_key(active):
    normal = features.validate_active(active)
    key = min(normal, features.reflect_active(normal))
    return hashlib.sha256(b"".join(value.to_bytes(2, "little") for value in key)).hexdigest()


def load_games(path):
    games = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len(games) != 512: raise ValueError("fresh pilot requires exactly 512 complete games")
    ids, roots, keys = set(), defaultdict(list), {}
    for game in games:
        if (game.get("schema") != "papersoccer.top-three.training-game.v1"
                or game.get("training_eligible") is not True or game.get("source_kind") != "fresh-generated"
                or game.get("rule_terminal") is not True or game.get("split") not in ("train", "validation")
                or type(game.get("winner")) is not int or game.get("winner") not in (0, 1)
                or type(game.get("candidate_player")) is not int or game.get("candidate_player") not in (0, 1)
                or type(game.get("seed")) is not int or game["seed"] < 0):
            raise ValueError("game is not an eligible fresh training trajectory")
        for key in ("game_id", "root_group_id", "opponent"):
            value = game.get(key)
            if not isinstance(value, str) or not value or any(ord(character) < 32 or ord(character) > 126 for character in value):
                raise ValueError("game identity must be printable ASCII")
        if set(game.get("source_sha256", {})) != {"candidate", "opponent"} or any(
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                for value in game["source_sha256"].values()): raise ValueError("game source identities absent")
        profiles = game.get("search_profiles", {})
        if set(profiles) != {"candidate", "opponent"} or any(
                not isinstance(value, list) or len(value) != 2
                or any(type(clock) is not int or clock < 1 for clock in value)
                or value[0] > 1000 or value[1] > 200 for value in profiles.values()):
            raise ValueError("game generation clock profiles invalid")
        if game["game_id"] in ids: raise ValueError("duplicate generated game")
        ids.add(game["game_id"])
        root = experiments.state(game["root_transcript"])
        key = experiments.fingerprint(root)
        if key != game["root_state_sha256"] or root.winner is not None or game.get("root_drawn_edges") != len(root.used_segments):
            raise ValueError("root identity changed")
        if key in keys and keys[key] != game["root_group_id"]: raise ValueError("canonical root has two group identities")
        keys[key] = game["root_group_id"]
        prefix = game["root_transcript"]
        if prefix and not game["transcript"].startswith(prefix + "/"): raise ValueError("game does not extend its root")
        terminal = experiments.state(game["transcript"])
        if terminal.winner != game["winner"]: raise ValueError("game does not reach declared terminal")
        roots[game["root_group_id"]].append(game)
    for paired in roots.values():
        if (len(paired) != 2 or {game["candidate_player"] for game in paired} != {0, 1}
                or len({game["split"] for game in paired}) != 1
                or len({game["root_state_sha256"] for game in paired}) != 1
                or len({game["opponent"] for game in paired}) != 1
                or len({json.dumps(game["source_sha256"], sort_keys=True) for game in paired}) != 1):
            raise ValueError("root pair colors/split/source identity changed")
    split_roots = Counter(paired[0]["split"] for paired in roots.values())
    if split_roots != {"train": 128, "validation": 128}: raise ValueError("pilot root quota changed")
    if len({game["opponent"] for game in games}) < 4: raise ValueError("pilot requires a multi-opponent distribution")
    return games


def sample_positions(games):
    positions = []
    # Train-first precedence is fixed before labels/metrics are available.
    for game in sorted(games, key=lambda row: (row["split"] != "train", row["game_id"])):
        actions = game["transcript"].split("/")
        start = len(game["root_transcript"].split("/")) if game["root_transcript"] else 0
        state = features.ReplayState()
        candidates = []
        for turn, action in enumerate(actions):
            if turn >= start and state.winner is None:
                prefix = "/".join(actions[:turn])
                active = features.encode_active(state)
                candidates.append({"position_id": game["game_id"] + ":" + str(turn),
                    "game_id": game["game_id"], "root_group_id": game["root_group_id"],
                    "group_id": game["root_group_id"], "source": "top-three-fresh-generated",
                    "split": game["split"], "winner": game["winner"], "mover": state.to_move,
                    "prefix": prefix, "edges": len(state.used_segments), "parent_active": list(active),
                    "canonical_state": experiments.fingerprint(state), "canonical_features": feature_key(active)})
            features.apply_complete_turn(state, state.to_move, action)
        early = [row for row in candidates if row["edges"] <= 12][:4]
        later = [row for row in candidates if row["edges"] > 12]
        sampled_later = [later[index * (len(later) - 1) // 7] for index in range(8)] if len(later) > 8 else later
        positions.extend(early + sampled_later)
    unique, states, active = [], set(), set()
    for row in positions:
        if row["canonical_state"] in states or row["canonical_features"] in active: continue
        states.add(row["canonical_state"])
        active.add(row["canonical_features"])
        unique.append(row)
    return unique


def prepare(games_path, teacher_path, binary, output, nodes):
    if nodes not in (64000, 256000): raise ValueError("pilot teacher budget must be 64k or 256k")
    identity = record(teacher_path)
    if identity["sha256"] != TEACHER_SHA256: raise ValueError("accepted teacher changed")
    games = load_games(games_path)
    positions = sample_positions(games)
    output.mkdir(parents=True, exist_ok=True)
    content = HEADER + "".join("\t".join(str(row[key]) for key in ("position_id", "root_group_id", "group_id", "source", "split", "winner", "mover", "prefix")) + "\n" for row in positions)
    once(output / "positions.tsv", content.encode("ascii"))
    body = {"schema": SCHEMA, "kind": "label-preparation", "games": record(games_path), "teacher": identity,
            "native_teacher": record(binary), "positions_file": record(output / "positions.tsv"),
            "positions": positions, "nodes": nodes, "source": record(__file__),
            "native_validator": record(corpus.__file__), "labeling_started": False}
    bundle = hashlib.sha256(representation.canonical(body)).hexdigest()
    command = [str(Path(binary).resolve()), "--model", identity["path"], "--model-sha256", TEACHER_SHA256,
               "--campaign-id", CAMPAIGN_ID, "--tree-nodes", str(nodes), "--time-ms", "0",
               "--max-actions", "250", "--max-partial-paths", "50000", "--exploration", "0.5", "--fpu", "0.5",
               "--emit-action-groups", "--source-bundle-body-sha256", bundle]
    document = {**body, "bundle_sha256": bundle, "command": command}
    once(output / "label-plan.json", representation.canonical(document))
    return {"plan": str(output / "label-plan.json"), "positions": len(positions),
            "early_validation_root_groups_before_labels": len({row["root_group_id"] for row in positions if row["split"] == "validation" and row["edges"] <= 12}),
            "labeling_started": False}


def import_labels(plan_path, labels_path, output, recovery_path=None):
    cpu_start = time.process_time()
    plan = json.loads(plan_path.read_bytes())
    if plan.get("schema") != SCHEMA or plan.get("kind") != "label-preparation": raise ValueError("label plan schema")
    body = {key: value for key, value in plan.items() if key not in ("bundle_sha256", "command")}
    if hashlib.sha256(representation.canonical(body)).hexdigest() != plan["bundle_sha256"]: raise ValueError("label preparation digest")
    for name in ("games", "teacher", "native_teacher", "positions_file", "source", "native_validator"): bound(plan[name])
    exclusions, prior_cpu_charge = set(), 0.0
    if recovery_path is not None:
        recovery = json.loads(recovery_path.read_bytes())
        if recovery.get("schema") != "papersoccer.compact-representation.supervision-recovery.v1" or recovery["base_plan"] != record(plan_path):
            raise ValueError("supervision recovery binding changed")
        exclusions = {row["position_id"] for row in recovery["excluded_positions"]}
        if recovery.get("policy") != "late-no-root-visits-v1" and exclusions != {"repr-root-0199-p0:55"}:
            raise ValueError("undeclared supervision exclusion")
        if recovery.get("policy") == "late-no-root-visits-v1" and recovery.get("complete_remainder_coverage") is not True:
            raise ValueError("prospective exclusion recovery is incomplete")
        for excluded in recovery["excluded_positions"]:
            original = next(row for row in plan["positions"] if row["position_id"] == excluded["position_id"])
            if original["canonical_state"] != excluded["canonical_state"]: raise ValueError("excluded state changed")
            if recovery.get("policy") == "late-no-root-visits-v1":
                failure = json.loads(bound(excluded["failure_receipt"]).read_bytes())
                error = bound(failure["stderr"]).read_text().splitlines()[0]
                if (original["edges"] <= 12 or failure["returncode"] != 2 or failure["identity"]["nodes"] != 64000
                        or failure["success"] is not False or original["position_id"] not in failure["identity"]["positions"]
                        or error != "jacek replay search teacher: " + original["position_id"] + ": unsolved search teacher completed no root visits"):
                    raise ValueError("exclusion falls outside the approved late64k predicate")
        aggregate = json.loads(labels_path.with_suffix(".manifest.json").read_bytes())
        if aggregate["recovery"] != record(recovery_path) or aggregate["labels"] != record(labels_path):
            raise ValueError("recovery aggregate binding changed")
        prior_cpu_charge = aggregate["charged_cpu_seconds"]
    positions = {row["position_id"]: row for row in plan["positions"] if row["position_id"] not in exclusions}
    output.parent.mkdir(parents=True, exist_ok=True)
    group_directory = output.parent / (output.stem + "-groups")
    group_directory.mkdir(exist_ok=True)
    observed, groups = set(), []
    training_keys, excluded, early_roots = set(), [], set()
    reached_validation, maximum_rss = False, 0
    with labels_path.open() as stream:
      for line in stream:
        native = corpus.validate_complete_turn_action_group(json.loads(line))
        group = native["group"]
        source = group["source_binding"]
        position = positions.get(source["position_id"])
        if (position is None or source["position_id"] in observed
                or native["source_bundle_body_sha256"] != plan["bundle_sha256"]
                or native["teacher"]["artifact_sha256"] != TEACHER_SHA256
                or group["work_budget"]["max_tree_nodes"] != plan["nodes"]
                or group["work_budget"]["max_time_ms"] != 0
                or source["campaign_id"] != CAMPAIGN_ID): raise ValueError("native label binding changed")
        observed.add(source["position_id"])
        prefix = "/".join(row["action"] for row in source["prefix"])
        if (prefix != position["prefix"] or any(source[key] != position[key] for key in ("root_group_id", "group_id", "split", "winner", "source"))
                or group["parent_mover"] != position["mover"]): raise ValueError("native label refers to another position")
        normalized = {"group_id": group["parent_identity"], "root_group_id": position["root_group_id"],
                       "game_id": position["game_id"], "split": position["split"], "mover": position["mover"],
                       "edges": position["edges"], "parent_active": position["parent_active"],
                       "canonical_state": position["canonical_state"], "exhaustive": group["successors_exhaustive"],
                       "teacher_value": group["root_value"], "successors": group["successors"],
                       "teacher": native["teacher"], "work_budget": group["work_budget"]}
        keys = {feature_key(position["parent_active"])} | {feature_key(row["active"]) for row in group["successors"]}
        if position["split"] == "train":
            if reached_validation: raise ValueError("label shards must retain frozen train-first order")
            training_keys.update(keys)
        else:
            reached_validation = True
            if keys & training_keys:
                excluded.append(group["parent_identity"])
                continue
        values = [row["teacher_value"] * (1 if row["value_mover"] == position["mover"] else -1) for row in group["successors"]]
        comparable_early = (position["split"] == "validation" and position["edges"] <= 12 and group["successors_exhaustive"]
                            and len(values) > 1 and max(values) - min(values) > 1e-4)
        if comparable_early: early_roots.add(position["root_group_id"])
        packed = gzip.compress(representation.canonical(normalized), mtime=0)
        group_path = group_directory / (group["parent_identity"] + ".json.gz")
        once(group_path, packed)
        groups.append({"group_id": group["parent_identity"], "root_group_id": position["root_group_id"],
                       "split": position["split"], "edges": position["edges"], "comparable_early": comparable_early,
                       "artifact": record(group_path)})
        maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        maximum_rss = maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
        if maximum_rss >= 12 * 1024**3: raise MemoryError("label aggregation exceeded 12 GiB")
        if prior_cpu_charge + time.process_time() - cpu_start >= 5395:
            once(output.parent / "aggregation-budget-stop.json", representation.canonical({"charged_cpu_seconds": prior_cpu_charge + time.process_time() - cpu_start, "cap": 5400}))
            raise RuntimeError("label aggregation reached the shared90CPUminute cap")
    if observed != set(positions): raise ValueError("native label set incomplete")
    result = {"schema": SCHEMA, "kind": "fresh-complete-turn-search-corpus", "teacher_runtime_sha256": TEACHER_SHA256,
              "label_plan": record(plan_path), "native_labels": record(labels_path), "group_artifacts": groups,
              "storage": "per-position-gzip-json-streamed-by-root", "peak_aggregation_rss_bytes": maximum_rss,
              "validation_overlap_groups_removed": len(excluded), "excluded_group_ids": excluded,
              "independent_comparable_early_validation_root_groups": len(early_roots),
              "production_training_admitted": len(early_roots) >= 100, "training_started": False}
    result.update(supervision_recovery=None if recovery_path is None else record(recovery_path),
                  excluded_position_ids=sorted(exclusions), importer_source=record(__file__),
                  charged_label_cpu_seconds=prior_cpu_charge + time.process_time() - cpu_start)
    output.parent.mkdir(parents=True, exist_ok=True)
    once(output, representation.canonical(result))
    return {key: value for key, value in result.items() if key not in ("group_artifacts", "excluded_group_ids")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare-labels")
    for name in ("games", "teacher", "native-teacher", "output"): prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--nodes", type=int, default=64000)
    ingest = subs.add_parser("import-labels")
    for name in ("plan", "labels", "output"): ingest.add_argument("--" + name, type=Path, required=True)
    ingest.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    if args.command == "prepare-labels": result = prepare(args.games, args.teacher, args.native_teacher, args.output, args.nodes)
    else: result = import_labels(args.plan, args.labels, args.output, args.recovery)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
