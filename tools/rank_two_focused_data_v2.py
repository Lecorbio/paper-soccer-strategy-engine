"""Frozen matched training trajectories and teacher-only corpus preparation."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import random
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tools import compact_representation_data as data
from tools import compact_representation_generate as generation
from tools import jacek_replay_corpus as labels
from tools import jacek_replay_features as rules
from tools import rank_two_focused_campaign_v1 as campaign
from tools import top_three_experiments as games

SCHEMA = campaign.SCHEMA + ".matched-data"
CAMPAIGN_ID = "rank-two-focused-network-20261007"


def check_current(plan, action, launch=False):
    from tools import rank_two_focused_work_v1 as work
    return work.checked(plan, action, launch=launch)


def freeze_generation(base, ancestry_path, output, used_percent):
    base, output = Path(base).resolve(), Path(output).resolve()
    current = campaign.read(base / "CURRENT.json")
    preliminary = dict(campaign_current=str(base / "CURRENT.json"),
                       activation=current["focused_activation"],
                       owner_thread_id=current["owner_thread_id"], used_percent=used_percent)
    check_current(preliminary, "generate", launch=True)
    if current.get("pending_exposures"):
        raise PermissionError("increment pending ancestry before drawing fresh roots")
    approval = campaign.read(campaign.verify(current["focused_plan"]))
    ancestry = campaign.read(ancestry_path)
    if ancestry.get("passed") is not True:
        raise ValueError("complete incremental ancestry required")
    prior = campaign.read(campaign.verify(ancestry["prior"]))
    source_database = campaign.verify(prior["inputs"]["database"])
    manifest = campaign.read(campaign.verify(approval["roster_manifest"]))
    if manifest["control"]["source"] != current["incumbent"] or len(manifest["opponents"]) != 8:
        raise ValueError("exact incumbent and frozen eight-opponent roster required")
    actors = {name: games.descriptor(campaign.verify(item["source"]), item["family"], [50, 10])
              for name, item in manifest["opponents"].items()}
    actors["rank_3"] = games.descriptor(campaign.verify(current["incumbent"]), "turn_action_v2", [50, 10])
    rng, seen, rows = random.Random(2026100710), set(), []
    with sqlite3.connect(source_database.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA cache_size=-16384")
        for index in range(256):
            for _ in range(10000):
                row = generation.fresh_root(rng, (4, 6, 8)[index % 3], seen)
                feature = data.feature_key(rules.encode_active(games.state(row["transcript"])))
                if feature in seen:
                    continue
                blocked = any(db.execute("SELECT 1 FROM keys WHERE category=? AND value=?",
                                         (category, key)).fetchone() is not None
                              for category, key in (("states", row["state_sha256"]), ("features", feature)))
                if blocked:
                    continue
                seen.add(feature)
                break
            else:
                raise ValueError("fresh training root draw exhausted; no seed replacement")
            group = f"focused-root-{index:04d}"
            rows.append({**row, "feature_sha256": feature, "root_group_id": group,
                         "cluster_id": group, "opponent": sorted(manifest["opponents"])[index % 8],
                         "split": "train" if index < 128 else "validation", "colors": [0, 1]})
    producers = {name: campaign.record(path) for name, path in dict(
        orchestrator=__file__, generator=generation.__file__, sampler=data.__file__,
        driver=games.__file__, rules=rules.__file__, worker=games.WORKER,
        lifecycle=campaign.ROOT / "tools/rank_two_focused_work_v1.py").items()}
    producers["controller"] = current["control_guard"]
    plan = dict(schema=SCHEMA + ".generation", **preliminary, seed=2026100710,
                candidate="rank_3", actors=actors, rows=rows, new_games=512,
                training_clocks_ms=[50, 10], strength_eligible=False, outcomes_as_targets=False,
                roster=approval["roster_manifest"], ancestry_ready=campaign.record(ancestry_path),
                producers=producers, maximum_workers=1, numerical_threads=1,
                maximum_game_wall_seconds=120, output=str(output),
                resource_budget=dict(wall_seconds=7200, cpu_capacity_seconds=28800,
                                     rss_bytes=2 * 1024**3, workers=3),
                split_roots=dict(Counter(row["split"] for row in rows)))
    campaign.immutable(output / "plan.json", plan)
    campaign.immutable(output / "EXPOSURE_PENDING.json", dict(
        schema=SCHEMA + ".reserved-training-roots", generation_plan=campaign.record(output / "plan.json"),
        training_eligible=False, before_next_fresh_bank=True))
    return plan


def completed_exposure(plan_path, output):
    plan = campaign.read(plan_path)
    output = Path(output)
    return campaign.immutable(output / "COMPLETED_EXPOSURE.json", dict(
        schema=SCHEMA + ".completed-training-trajectories", generation_plan=campaign.record(plan_path),
        result=campaign.record(output / "RESULT.json"), input=campaign.record(output / "envelopes.jsonl"),
        game_receipts=[campaign.record(output / "games" / (row["root_group_id"] + "-p" + str(color)) / "RESULT.json")
                       for row in plan["rows"] for color in row["colors"]],
        training_eligible=False, before_next_fresh_bank=True,
        games_retained=sum(len(row["colors"]) for row in plan["rows"])))


def generate(plan_path):
    plan = campaign.read(plan_path)
    check_current(plan, "generate", launch=True)
    if plan["schema"] != SCHEMA + ".generation" or plan["seed"] != 2026100710 or plan["candidate"] != "rank_3":
        raise ValueError("matched generation recipe differs")
    for reference in plan["producers"].values():
        campaign.verify(reference)
    output = Path(plan["output"])
    names = sorted(plan["actors"])
    builds = {name: games.build_worker(plan["actors"][name], output / "builds", "/usr/bin/clang++")
              for name in names}
    campaign.immutable(output / "builds.json", builds)
    envelopes, started = [], time.monotonic()
    for row in plan["rows"]:
        for color in row["colors"]:
            # Ownership/pause is checked at each boundary; an interrupted claim
            # remains spent and cannot be silently replayed.
            check_current(plan, "generate")
            identifier = row["root_group_id"] + "-p" + str(color)
            directory = output / "games" / identifier
            result_path = directory / "RESULT.json"
            if result_path.exists():
                envelope = campaign.read(result_path)
                if envelope["plan"] != campaign.record(plan_path):
                    raise ValueError("retained trajectory plan changed")
                games.training_trajectory(envelope)
            else:
                if (directory / "CLAIM.json").exists():
                    raise RuntimeError("unknown game claim retained; no automatic replay")
                campaign.immutable(directory / "CLAIM.json", dict(
                    plan=campaign.record(plan_path), row=row, color=color,
                    builds={name: builds[name] for name in ("rank_3", row["opponent"])}))
                begin = time.monotonic()
                def decision_guard():
                    if time.monotonic() - begin >= plan["maximum_game_wall_seconds"]:
                        raise RuntimeError("declared per-game wall cap reached")
                played = games.play(row, color, plan["actors"]["rank_3"], plan["actors"][row["opponent"]],
                                    builds["rank_3"], builds[row["opponent"]], decision_guard=decision_guard)
                envelope = dict(plan=campaign.record(plan_path), game_id=identifier, seed=plan["seed"],
                                split=row["split"], opponent_name=row["opponent"],
                                candidate=plan["actors"]["rank_3"], opponent=plan["actors"][row["opponent"]],
                                root=dict(source_kind="fresh-generated", root_group_id=row["root_group_id"],
                                          transcript=row["transcript"]), game=played,
                                generation_wall_seconds=time.monotonic() - begin)
                campaign.immutable(result_path, envelope)
                campaign.immutable(directory / "EXPOSURE.json", dict(
                    schema=SCHEMA + ".played-trajectory-exposure", input=campaign.record(result_path),
                    generation_plan=campaign.record(plan_path), training_eligible=False,
                    before_next_fresh_bank=True, includes_partial=played["failure"] is not None))
                if played["failure"] is not None:
                    raise RuntimeError("training trajectory operational failure; exact receipt retained")
                games.training_trajectory(envelope)
            envelopes.append(envelope)
            campaign.atomic(output / "PROGRESS.json", dict(games_completed=len(envelopes),
                            wall_seconds=time.monotonic() - started, total_games=512))
    envelope_path = output / "envelopes.jsonl"
    campaign.immutable(envelope_path, b"".join(campaign.canonical(row) for row in envelopes))
    exported = games.export_training(envelope_path, output / "games.jsonl")
    data.load_games(output / "games.jsonl")
    result = dict(passed=True, plan=campaign.record(plan_path), games=512,
                  trajectories=campaign.record(output / "games.jsonl"), export=exported,
                  wall_seconds=time.monotonic() - started, strength_eligible=False)
    campaign.immutable(output / "RESULT.json", result)
    completed_exposure(plan_path, output)
    return result


def prepare_labels(generation_plan, games_path, teacher_path, native_path, output):
    plan = campaign.read(generation_plan)
    check_current(plan, "label", launch=True)
    teacher = data.record(teacher_path)
    if teacher["sha256"] != data.TEACHER_SHA256:
        raise ValueError("accepted teacher changed")
    trajectories = data.load_games(games_path)
    if any(row["seed"] != 2026100710 or row["source_sha256"]["candidate"] != campaign.BASELINE
           for row in trajectories):
        raise ValueError("fresh incumbent generation binding differs")
    positions = data.sample_positions(trajectories)
    if len(positions) > 6144:
        raise ValueError("bounded sampler quota exceeded")
    output = Path(output)
    payload = data.HEADER + "".join("\t".join(str(row[key]) for key in
        ("position_id", "root_group_id", "group_id", "source", "split", "winner", "mover", "prefix"))
        + "\n" for row in positions)
    campaign.immutable(output / "positions.tsv", payload.encode("ascii"))
    body = dict(schema=SCHEMA + ".teacher-plan", generation_plan=campaign.record(generation_plan),
                games=data.record(games_path), teacher=teacher, native_teacher=data.record(native_path),
                positions_file=data.record(output / "positions.tsv"), positions=positions, nodes=64000,
                campaign_id=CAMPAIGN_ID, outcomes_as_targets=False,
                producer=campaign.record(__file__), validator=campaign.record(labels.__file__),
                lifecycle=campaign.record(campaign.ROOT / "tools/rank_two_focused_work_v1.py"),
                activation=plan["activation"], owner_thread_id=plan["owner_thread_id"],
                campaign_current=plan["campaign_current"], used_percent=plan["used_percent"],
                maximum_position_cpu_seconds=120, maximum_workers=10,
                maximum_rss_bytes=18 * 1024**3, numerical_threads=1)
    import hashlib
    body_sha = hashlib.sha256(campaign.canonical(body)).hexdigest()
    command = [str(Path(native_path).resolve()), "--model", teacher["path"], "--model-sha256", data.TEACHER_SHA256,
               "--campaign-id", CAMPAIGN_ID, "--tree-nodes", "64000", "--time-ms", "0",
               "--max-actions", "250", "--max-partial-paths", "50000", "--exploration", "0.5", "--fpu", "0.5",
               "--emit-action-groups", "--source-bundle-body-sha256", body_sha]
    document = {**body, "bundle_sha256": body_sha, "command": command}
    campaign.immutable(output / "PLAN.json", document)
    return document


def normalized_label(native, position, plan):
    native = labels.validate_complete_turn_action_group(native)
    group, source = native["group"], native["group"]["source_binding"]
    if (native["source_bundle_body_sha256"] != plan["bundle_sha256"]
            or native["teacher"]["artifact_sha256"] != data.TEACHER_SHA256
            or group["work_budget"]["max_tree_nodes"] != 64000
            or group["work_budget"]["max_time_ms"] != 0
            or source["campaign_id"] != CAMPAIGN_ID):
        raise ValueError("teacher/budget/recipe binding differs")
    prefix = "/".join(row["action"] for row in source["prefix"])
    if (prefix != position["prefix"] or group["parent_mover"] != position["mover"]
            or any(source[key] != position[key] for key in
                   ("position_id", "root_group_id", "group_id", "split", "winner", "source"))):
        raise ValueError("label is not bound to the exact originally played parent")
    # Outcome fields stay in provenance only. All targets remain the exact
    # native teacher/proof/terminal targets, with their native mover signs.
    return dict(group_id=group["parent_identity"], root_group_id=position["root_group_id"],
                game_id=position["game_id"], split=position["split"], mover=position["mover"],
                edges=position["edges"], parent_active=position["parent_active"],
                canonical_state=position["canonical_state"], exhaustive=group["successors_exhaustive"],
                teacher_value=group["root_value"], successors=group["successors"],
                teacher=native["teacher"], work_budget=group["work_budget"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.plan)))


if __name__ == "__main__":
    main()
