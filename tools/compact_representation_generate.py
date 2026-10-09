#!/usr/bin/env python3
"""Source-bound fresh training games and a CPU-bounded search-teacher probe."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import time

for name in ("MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
import compact_representation as representation
import compact_representation_data as data
import jacek_replay_corpus as corpus
import jacek_replay_features as rules
import top_three_campaign as campaign
import top_three_experiments as driver

SCHEMA = "papersoccer.compact-representation.generation.v1"
SEED = 2026091604


def cpu():
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return usage.ru_utime + usage.ru_stime


def peak_bytes():
    value = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def emit(path, body):
    campaign.immutable(Path(path), representation.canonical(body))


def fresh_root(rng, depth, seen):
    for _ in range(10000):
        state = rules.ReplayState()
        transcript = []
        while len(state.used_segments) < depth and state.winner is None:
            mover, action = state.to_move, ""
            while state.winner is None and state.to_move == mover:
                options = []
                for direction in range(8):
                    child = copy.deepcopy(state)
                    try: rules.apply_primitive(child, direction)
                    except ValueError: continue
                    options.append((direction, child))
                if not options: break
                direction, state = options[rng.randrange(len(options))]
                action += str(direction)
            if not action: break
            transcript.append(action)
        if state.winner is not None or len(state.used_segments) != depth: continue
        key = driver.fingerprint(state)
        if key in seen: continue
        seen.add(key)
        return {"transcript": "/".join(transcript), "state_sha256": key, "drawn_edges": depth}
    raise RuntimeError("could not produce a fresh complete-turn root at the frozen depth")


def freeze(root, output, kind):
    path = output / "plan.json"
    if path.exists():
        saved = json.loads(path.read_bytes())
        for item in saved["producers"].values(): data.bound(item)
        for item in saved["actors"].values(): driver.verify(item["source"])
        if saved["kind"] != kind: raise ValueError("run kind changed")
        return saved
    roster = driver.roster(root)
    actors = {name: driver.descriptor(driver.verify(item["source"]), item["family"], [50, 10]) for name, item in roster.items()}
    rng = random.Random(SEED + (1 if kind == "throughput" else 0))
    seen, rows = set(), []
    names = list(campaign.CONTROLS)
    if kind == "throughput":
        for index, opponent in enumerate(("rank_4", "h62", "rank_4_jacek_hybrid", "neural_puct")):
            row = fresh_root(rng, (4, 6, 8)[index % 3], seen)
            rows.append({**row, "opponent": opponent, "cluster_id": f"throughput-root-{index:03d}",
                         "root_group_id": f"throughput-root-{index:03d}", "split": "train", "colors": [index % 2]})
    else:
        for index in range(256):
            row = fresh_root(rng, (4, 6, 8)[index % 3], seen)
            group = f"repr-root-{index:04d}"
            rows.append({**row, "opponent": names[index % len(names)], "cluster_id": group,
                         "root_group_id": group, "split": "train" if index < 128 else "validation", "colors": [0, 1]})
    producer_paths = {"orchestrator": Path(__file__), "driver": Path(driver.__file__),
                      "rules": Path(rules.__file__), "campaign": Path(campaign.__file__),
                      "worker": driver.WORKER}
    producers = {}
    for name, source in producer_paths.items():
        archive = output / "producers" / source.name
        campaign.immutable(archive, source.read_bytes())
        producers[name] = data.record(archive)
    plan = {"schema": SCHEMA, "kind": kind, "seed": SEED, "root_seed": SEED + (kind == "throughput"),
            "candidate": "compact_deployed", "actors": actors, "rows": rows,
            "training_clocks_ms": [50, 10], "strength_eligible": False, "max_workers": 1,
            "new_games": sum(len(row["colors"]) for row in rows),
            "root_depth_counts": dict(Counter(row["drawn_edges"] for row in rows)),
            "opponent_game_counts": dict(Counter(row["opponent"] for row in rows for _ in row["colors"])),
            "producers": producers, "actual_game_cpu_limit_seconds": 7200,
            "root": str(root.resolve())}
    emit(path, plan)
    return plan


def games(plan, output, compiler):
    build_root = Path(plan["root"]) / "representation-training-builds"
    names = {plan["candidate"], *(row["opponent"] for row in plan["rows"])}
    builds = {name: driver.build_worker(plan["actors"][name], build_root, compiler) for name in sorted(names)}
    emit(output / "builds.json", builds)
    completed, cpu_samples = [], []
    started = time.monotonic()
    for index, row in enumerate(plan["rows"]):
        for color in row["colors"]:
            if sum(cpu_samples) >= plan.get("actual_game_cpu_limit_seconds", 7200):
                raise RuntimeError("fresh-game generation reached its two CPU-hour limit")
            game_id = row["root_group_id"] + "-p" + str(color)
            path = output / "games" / (game_id + ".json")
            if path.exists():
                envelope = json.loads(path.read_bytes())
                driver.training_trajectory(envelope)
            else:
                begin_cpu, begin = cpu(), time.monotonic()
                played = driver.play(row, color, plan["actors"][plan["candidate"]], plan["actors"][row["opponent"]],
                                     builds[plan["candidate"]], builds[row["opponent"]])
                envelope = {"game_id": game_id, "seed": plan["root_seed"], "split": row["split"],
                            "opponent_name": row["opponent"], "candidate": plan["actors"][plan["candidate"]],
                            "opponent": plan["actors"][row["opponent"]],
                            "root": {"source_kind": "fresh-generated", "root_group_id": row["root_group_id"], "transcript": row["transcript"]},
                            "game": played, "generation_cpu_seconds": cpu() - begin_cpu,
                            "generation_wall_seconds": time.monotonic() - begin}
                if played["failure"] is not None:
                    emit(output / "failures" / (game_id + ".json"), envelope)
                    raise RuntimeError(f"generated game failed: {game_id}: {played['failure']}")
                driver.training_trajectory(envelope)
                emit(path, envelope)
            completed.append(envelope)
            cpu_samples.append(envelope["generation_cpu_seconds"])
        if (index + 1) % 8 == 0 or plan["kind"] == "throughput":
            progress = {"root_groups_completed": index + 1, "games_completed": len(completed), "cpu_seconds": sum(cpu_samples),
                        "wall_seconds": time.monotonic() - started, "peak_children_rss_bytes": peak_bytes()}
            (output / "status.json").write_bytes(representation.canonical(progress))
            print(json.dumps(progress), flush=True)
    content = b"".join(representation.canonical(envelope) for envelope in completed)
    emit_path = output / "envelopes.jsonl"
    campaign.immutable(emit_path, content)
    exported = driver.export_training(emit_path, output / "games.jsonl")
    report = {"schema": SCHEMA, "games": len(completed), "cpu_seconds": sum(cpu_samples),
              "maximum_game_cpu_seconds": max(cpu_samples), "wall_seconds": time.monotonic() - started,
              "peak_children_rss_bytes": peak_bytes(), "export": exported,
              "projected_512_cpu_seconds_conservative": max(cpu_samples) * 512 * 2,
              "cpu_projection_safety_factor": 2, "training_clocks_ms": [50, 10], "strength_eligible": False}
    emit(output / "generation.json", report)
    return completed, report


def probe_positions(envelopes):
    buckets = {"early": [], "middle": [], "late": []}
    seen = set()
    for envelope in envelopes:
        trajectory = driver.training_trajectory(envelope)
        state = rules.ReplayState()
        actions = trajectory["transcript"].split("/")
        start = len(trajectory["root_transcript"].split("/"))
        for turn, action in enumerate(actions):
            if turn >= start:
                key = driver.fingerprint(state)
                if key not in seen:
                    seen.add(key)
                    depth = len(state.used_segments)
                    name = "early" if depth <= 12 else "middle" if depth < 40 else "late"
                    buckets[name].append({"position_id": trajectory["game_id"] + ":" + str(turn),
                        "root_group_id": trajectory["root_group_id"], "group_id": trajectory["root_group_id"],
                        "source": "fresh-throughput", "split": "train", "winner": trajectory["winner"],
                        "mover": state.to_move, "prefix": "/".join(actions[:turn]), "stratum": name,
                        "drawn_edges": depth})
            rules.apply_complete_turn(state, state.to_move, action)
    chosen = []
    while len(chosen) < 16 and any(buckets.values()):
        for name in ("early", "middle", "late"):
            if buckets[name] and len(chosen) < 16: chosen.append(buckets[name].pop(len(buckets[name]) // 2))
    return chosen


def label_probe(positions, output, native, teacher):
    if data.record(teacher)["sha256"] != data.TEACHER_SHA256: raise ValueError("accepted teacher changed")
    freeze_body = {"schema": SCHEMA, "kind": "bounded-teacher-throughput", "native": data.record(native),
                   "teacher": data.record(teacher), "nodes": 64000, "cpu_limit_seconds": 120,
                   "requested_positions": positions, "games": data.record(output / "games.jsonl")}
    digest = campaign.digest(representation.canonical(freeze_body))
    emit(output / "label-probe-plan.json", {**freeze_body, "body_sha256": digest})
    begin_cpu, begin_wall = cpu(), time.monotonic()
    reports = []
    for index, position in enumerate(positions):
        remaining = 120 - (cpu() - begin_cpu)
        if remaining < 2: break
        directory = output / "labels" / f"{index:02d}"
        directory.mkdir(parents=True, exist_ok=True)
        command = [str(native.resolve()), "--model", str(teacher.resolve()), "--model-sha256", data.TEACHER_SHA256,
                   "--campaign-id", "top-three-representation-throughput-20260916", "--tree-nodes", "64000", "--time-ms", "0",
                   "--max-actions", "250", "--max-partial-paths", "50000", "--exploration", "0.5", "--fpu", "0.5",
                   "--emit-action-groups", "--source-bundle-body-sha256", digest]
        payload = data.HEADER + "\t".join(str(position[key]) for key in ("position_id", "root_group_id", "group_id", "source", "split", "winner", "mover", "prefix")) + "\n"
        emit(directory / "claim.json", {"position": position, "command": command, "remaining_cpu_seconds": remaining})
        before_cpu, started = cpu(), time.monotonic()
        limit = max(1, math.floor(remaining))
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CPU, (limit, limit)))
        try:
            stdout, stderr = process.communicate(payload.encode(), timeout=remaining + 2)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        elapsed, used_cpu = time.monotonic() - started, cpu() - before_cpu
        (directory / "stdout.jsonl").write_bytes(stdout)
        (directory / "stderr.txt").write_bytes(stderr)
        row = {"position_id": position["position_id"], "stratum": position["stratum"], "drawn_edges": position["drawn_edges"],
               "cpu_seconds": used_cpu, "wall_seconds": elapsed, "returncode": process.returncode,
               "peak_children_rss_bytes": peak_bytes(), "teacher_nodes": 64000, "valid_label": False}
        if process.returncode == 0:
            native_row = corpus.validate_complete_turn_action_group(json.loads(stdout))
            if native_row["teacher"]["artifact_sha256"] != data.TEACHER_SHA256: raise ValueError("teacher probe runtime drift")
            row.update(valid_label=True, successors=len(native_row["group"]["successors"]),
                       successors_exhaustive=native_row["group"]["successors_exhaustive"])
        reports.append(row)
        emit(directory / "result.json", row)
        print(json.dumps({"teacher_probe_completed": len(reports), **row}), flush=True)
        if not row["valid_label"]: break
    valid = [row for row in reports if row["valid_label"]]
    result = {"schema": SCHEMA, "stage": "throughput-probe-complete", "requested_groups": len(positions),
              "completed_groups": len(valid), "group_reports": reports, "cpu_seconds": cpu() - begin_cpu,
              "wall_seconds": time.monotonic() - begin_wall, "peak_children_rss_bytes": peak_bytes(),
              "budget_unchanged_nodes": 64000, "independent_source_games": 4, "qualification_eligible": False}
    if valid:
        mean = sum(row["cpu_seconds"] for row in valid) / len(valid)
        maximum = max(row["cpu_seconds"] for row in valid)
        result.update(mean_cpu_seconds_per_group=mean, maximum_cpu_seconds_per_group=maximum,
                      maximum_positions_for_512_games=6144, projected_6144_group_cpu_seconds=mean * 6144,
                      conservative_6144_group_cpu_seconds=maximum * 6144)
    emit(output / "teacher-throughput.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe", "generate"))
    parser.add_argument("--root", type=Path, default=campaign.DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="/usr/bin/clang++")
    parser.add_argument("--native-teacher", type=Path, default=campaign.ROOT / "build/top_three/papersoccer_jacek_replay_search_teacher")
    parser.add_argument("--probe-report", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        if args.probe_report is None: parser.error("generation requires the measured four-game report")
        probe = json.loads(args.probe_report.read_bytes())
        if (probe.get("schema") != SCHEMA or probe.get("games") != 4
                or probe.get("training_clocks_ms") != [50, 10]
                or not 0 < probe.get("projected_512_cpu_seconds_conservative", 0) <= 7200):
            raise ValueError("fresh-game projected cost exceeds authorized two CPU-hours or lacks measured evidence")
        emit(args.output / "cost-admission.json", {"probe": data.record(args.probe_report), "maximum_cpu_seconds": 7200})
    plan = freeze(args.root, args.output, "throughput" if args.command == "probe" else "production-training")
    envelopes, generation = games(plan, args.output, args.compiler)
    if args.command == "probe":
        labels = label_probe(probe_positions(envelopes), args.output, args.native_teacher, args.root / "controls/teacher/model.runtime")
        print(json.dumps({"generation": generation, "labels": labels}), flush=True)
    else: print(json.dumps(generation), flush=True)


if __name__ == "__main__": main()
