#!/usr/bin/env python3
"""Source-bound, paired actual-clock experiments on fresh development roots.

State games use independently compiled standalone submissions. Empty-board
protocol games use the unchanged process referee. Smoke clocks are explicitly
ineligible for strength claims. Protected campaign banks are never discovered.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import selectors
import shutil
import subprocess
import threading
import time

try:
    from . import top_three_campaign as campaign, jacek_replay_features as rules
except ImportError:
    import top_three_campaign as campaign
    import jacek_replay_features as rules

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tools/top_three_state_worker.cpp"
SCHEMA = "papersoccer.top-three.experiments.v1"
RUN_SCHEMA = "papersoccer.top-three.experiments.run.v2"
MANY_RUN_SCHEMA = "papersoccer.top-three.experiments.run-many.v1"
MANY_ROOT_SCHEMA = "papersoccer.top-three.experiments.shared-control-root.v1"
MANY_RESULT_SCHEMA = "papersoccer.top-three.experiments.shared-control-result.v1"
DEPLOYMENT_PREFIX_CLOCK = "deployment-prefix-own-decisions-v1"
FRESH_ROOT_CLOCK = "fresh-root-own-decisions-v1"
FAMILIES = {"turn_action_v2": 0, "compact": 1, "h62": 2, "neural_puct": 3, "opening_book": 4}
ROSTER_FAMILIES = {name: ("compact" if name == "compact_deployed" else "h62" if name == "h62"
                       else "neural_puct" if name == "neural_puct" else "turn_action_v2") for name in campaign.CONTROLS}
ACTIVE = set()
ACTIVE_LOCK = threading.Lock()
RSS_LIMIT = 12 * 1024**3


def raw_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": campaign.digest(path.read_bytes())}


def verify(item):
    path = Path(item["path"])
    if raw_record(path) != item:
        raise ValueError("source/binary identity mismatch")
    return path


def emit(path, value):
    campaign.immutable(Path(path), campaign.canonical(value))


def read(path):
    return json.loads(Path(path).read_text())


def load_snapshot(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path), "sha256": campaign.digest(raw)}


def descriptor(source, family, clocks):
    if family not in FAMILIES or len(clocks) != 2 or any(type(x) is not int or x < 1 for x in clocks) or clocks[0] > 1000 or clocks[1] > 200:
        raise ValueError("invalid family or explicit clocks")
    has_book = b"bool try_opening_book(" in Path(source).read_bytes()
    if has_book != (family == "opening_book"):
        raise ValueError("opening-book source requires explicit opening_book family, and that family requires its hook")
    return {"source": raw_record(source), "family": family, "clocks_ms": clocks}


def roster(root):
    manifest = campaign.verify(root)
    return {name: descriptor(campaign.verify_record(root, entry), ROSTER_FAMILIES[name], entry["clocks_ms"])
            for name, entry in manifest["controls"].items()}


def state(prefix):
    result = rules.ReplayState()
    if prefix not in ("", "-"):
        for action in prefix.split("/"):
            rules.apply_complete_turn(result, result.to_move, action)
    return result


def prior_decision_counts(prefix, clock_policy=DEPLOYMENT_PREFIX_CLOCK):
    if clock_policy == FRESH_ROOT_CLOCK:
        return [0, 0]
    if clock_policy != DEPLOYMENT_PREFIX_CLOCK:
        raise ValueError("unknown state-game clock policy")
    counts = [0, 0]
    position = rules.ReplayState()
    if prefix not in ("", "-"):
        for action in prefix.split("/"):
            mover = position.to_move
            rules.apply_complete_turn(position, mover, action)
            counts[mover] += 1
    return counts


def compiler_signature(compiler):
    compiler = str(Path(shutil.which(compiler) or compiler).resolve())
    return {"executable": raw_record(compiler),
            "version": subprocess.check_output([compiler, "--version"], text=True)}


def verify_build_receipt(build, actor, signature):
    if build["descriptor"] != actor or build["compiler_binary"] != signature["executable"] or build["compiler"] != signature["version"]:
        raise ValueError("worker source/compiler build binding changed")
    for key in ("binary", "compiled_source", "worker_source", "compiler_binary"):
        verify(build[key])
    if build["compiled_source"]["sha256"] != actor["source"]["sha256"]:
        raise ValueError("compiled source differs from requested actor")
    if build["worker_source"]["sha256"] != campaign.digest(WORKER.read_bytes()):
        raise ValueError("compiled worker wrapper differs from current wrapper")
    receipt_path = Path(build["binary"]["path"]).parent / "build.json"
    if read(receipt_path) != build:
        raise ValueError("worker build receipt differs from in-memory identity")
    return {"receipt": raw_record(receipt_path), "binary": build["binary"]}


def verify_resume_builds(plan, signature, expected_schema=RUN_SCHEMA):
    if plan.get("schema") != expected_schema or plan.get("clock_policy") != DEPLOYMENT_PREFIX_CLOCK:
        raise ValueError("legacy/different clock policy requires a fresh run namespace")
    if plan.get("compiler") != signature:
        raise ValueError("cannot resume with a different compiler")
    if not plan.get("worker_builds"):
        raise ValueError("run omits frozen worker builds")
    for binding in plan["worker_builds"].values():
        build = read(verify(binding["receipt"]))
        if build["binary"] != binding["binary"]:
            raise ValueError("run worker binary differs from its build receipt")
        verify(binding["binary"])
        for key in ("compiled_source", "worker_source", "compiler_binary"):
            verify(build[key])


def fingerprint(value):
    # Exact physical state with mover-relative rotation and reflection. Unlike
    # a feature-vector hash this preserves ball and every used edge exactly.
    choices = []
    for reflected in (False, True):
        def point(p):
            p = rules.rotate_point(p) if value.to_move else p
            return rules.reflect_point(p) if reflected else p
        edges = sorted(tuple(sorted((point(a), point(b)))) for a, b in value.used_segments)
        visited = sorted(point(p) for p, count in value.visit_count.items() if count)
        choices.append(campaign.canonical([point(value.ball), edges, visited, value.winner]))
    return campaign.digest(min(choices))


def freeze_bank(root, input_path, output, stage, excluded_banks=()):
    names = set(roster(root))
    pairs = {"screen": 32, "confirmation": 100, "smoke": 1}[stage]
    excluded_states = set()
    excluded_clusters = set()
    exclusions = []
    for path in excluded_banks:
        old = read(path)
        excluded_states.update(row["state_sha256"] for row in old["rows"])
        excluded_clusters.update(row["cluster_id"] for row in old["rows"])
        exclusions.append(raw_record(path))
    if stage == "confirmation" and not excluded_banks:
        raise ValueError("confirmation requires explicit screen-bank exclusion")
    rows = []
    seen = set()
    for raw in Path(input_path).read_text().splitlines():
        item = json.loads(raw)
        if set(item) != {"opponent", "transcript", "cluster_id"} or item["opponent"] not in names or not isinstance(item["cluster_id"], str) or not item["cluster_id"]:
            raise ValueError("root rows require opponent/transcript/cluster_id")
        position = state(item["transcript"])
        if not position.used_segments or position.winner is not None:
            raise ValueError("state banks need live nonempty roots; use process referee for empty board")
        key = fingerprint(position)
        if key in seen or key in excluded_states or item["cluster_id"] in excluded_clusters:
            raise ValueError("duplicate/previously exposed canonical root or cluster")
        seen.add(key)
        rows.append({**item, "state_sha256": key, "drawn_edges": len(position.used_segments)})
    counts = collections.Counter(r["opponent"] for r in rows)
    if set(counts) != names or set(counts.values()) != {pairs}:
        raise ValueError(f"{stage} requires {pairs} roots for each of seven opponents")
    value = {"schema": SCHEMA, "kind": "bank", "stage": stage, "pairs_per_opponent": pairs,
             "input": raw_record(input_path), "exclusions": exclusions, "rows": rows,
             "canonical_rules": raw_record(rules.__file__), "protected_data": False}
    validate_bank(value, names)
    emit(output, value)
    return value


def validate_bank(bank, names):
    if bank.get("schema") != SCHEMA or bank.get("kind") != "bank" or bank.get("protected_data") is not False:
        raise ValueError("invalid unprotected bank contract")
    stage = bank["stage"]
    pairs = {"screen": 32, "confirmation": 100, "smoke": 1}[stage]
    if bank["pairs_per_opponent"] != pairs or raw_record(rules.__file__) != bank["canonical_rules"]:
        raise ValueError("bank pair-count or rule-source mismatch")
    excluded_states, excluded_clusters = set(), set()
    for item in bank["exclusions"]:
        old = read(verify(item))
        excluded_states.update(r["state_sha256"] for r in old["rows"])
        excluded_clusters.update(r["cluster_id"] for r in old["rows"])
    if stage == "confirmation" and not bank["exclusions"]:
        raise ValueError("confirmation lacks prior-bank exclusion")
    seen = set()
    counts = collections.Counter()
    clusters = collections.defaultdict(set)
    depths = collections.defaultdict(collections.Counter)
    for row in bank["rows"]:
        position = state(row["transcript"])
        key = fingerprint(position)
        if (not position.used_segments or position.winner is not None or key != row["state_sha256"] or
            len(position.used_segments) != row["drawn_edges"] or key in seen or key in excluded_states or
            row["cluster_id"] in excluded_clusters):
            raise ValueError("bank root duplicate, overlap, terminal, or identity mismatch")
        seen.add(key)
        counts[row["opponent"]] += 1
        clusters[row["opponent"]].add(row["cluster_id"])
        depths[row["opponent"]][row["drawn_edges"]] += 1
    if set(counts) != set(names) or set(counts.values()) != {pairs}:
        raise ValueError("bank does not match seven-opponent paired contract")
    if any(len(values) != pairs for values in clusters.values()):
        raise ValueError("each opponent requires independent root cluster IDs")
    if stage != "smoke" and any(dict(values) != {8: pairs // 4, 12: pairs // 4, 20: pairs // 4, 40: pairs // 4} for values in depths.values()):
        raise ValueError("bank must balance 8/12/20/40 drawn-edge strata")
    if bank.get("generator_schema") is not None:
        try:
            from . import top_three_development_banks as generated_banks
        except ImportError:
            import top_three_development_banks as generated_banks
        generated_banks.validate_generated_contract(bank, names)


def build_worker(item, directory, compiler):
    compiler = shutil.which(compiler) or compiler
    source = verify(item["source"])
    descriptor(source, item['family'], item['clocks_ms'])
    compiler_identity = raw_record(compiler)
    worker_bytes = WORKER.read_bytes()
    source_bytes = source.read_bytes()
    if campaign.digest(source_bytes) != item["source"]["sha256"]:
        raise ValueError("source changed before compile snapshot")
    key = campaign.digest(campaign.canonical(["archived-build-v2", item, campaign.digest(worker_bytes), compiler_identity]))
    directory = Path(directory).resolve() / key
    receipt = directory / "build.json"
    if receipt.exists():
        saved = read(receipt)
        verify_build_receipt(saved, item, compiler_signature(compiler))
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    archived_worker, archived_source = directory / "worker.cpp", directory / "submission.cpp"
    campaign.immutable(archived_worker, worker_bytes)
    campaign.immutable(archived_source, source_bytes)
    binary = directory / "worker"
    command = [compiler, "-std=c++20", "-O3", "-DNDEBUG", f'-DTOP_THREE_SOURCE="{archived_source}"',
               f'-DTOP_THREE_FAMILY={FAMILIES[item["family"]]}', str(archived_worker), "-o", str(binary)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    if result.returncode:
        raise RuntimeError(result.stderr)
    saved = {"descriptor": item, "worker_source": raw_record(archived_worker), "compiled_source": raw_record(archived_source), "binary": raw_record(binary),
             "compiler_binary": compiler_identity,
             "compiler": subprocess.check_output([compiler, "--version"], text=True), "command": command}
    emit(receipt, saved)
    return saved


def rss_guard():
    with ACTIVE_LOCK:
        pids = sorted(ACTIVE)
    if not pids:
        return
    result = subprocess.run(["ps", "-o", "rss=", "-p", ",".join(map(str, pids))], capture_output=True, text=True)
    rss = sum(int(x) for x in result.stdout.split()) * 1024
    if rss > RSS_LIMIT:
        raise RuntimeError("aggregate owned-worker RSS exceeded 12 GiB")


class StateWorker:
    def __init__(self, build):
        binary = verify(build["binary"])
        self.child = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, bufsize=0)
        os.set_blocking(self.child.stdout.fileno(), False)
        with ACTIVE_LOCK:
            ACTIVE.add(self.child.pid)

    def close(self):
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait()
        with ACTIVE_LOCK:
            ACTIVE.discard(self.child.pid)
        self.child.stdin.close()
        self.child.stdout.close()

    def choose(self, prefix, budget, external, watchdog_seconds=3.0):
        start = time.monotonic()
        self.child.stdin.write(f"{budget}\t{prefix or '-'}\n".encode())
        self.child.stdin.flush()
        data = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.child.stdout, selectors.EVENT_READ)
            # State reconstruction/IPC is excluded from engine timing but
            # bounded by a watchdog. Every engine decision still meets1000/200.
            while b"\n" not in data:
                if time.monotonic() - start > watchdog_seconds:
                    raise RuntimeError("state worker watchdog")
                if selector.select(timeout=0.1):
                    chunk = os.read(self.child.stdout.fileno(), 65537 - len(data))
                    if not chunk:
                        raise RuntimeError("worker closed output without complete response")
                    data.extend(chunk)
                    if len(data) > 65536:
                        raise RuntimeError("worker response exceeds 64 KiB")
                else:
                    rss_guard()
        if data.count(b"\n") != 1 or not data.endswith(b"\n"):
            raise RuntimeError("worker emitted extra response bytes")
        fields = data.decode("ascii").rstrip("\n").split("\t")
        if len(fields) != 2:
            raise RuntimeError("worker crash or malformed result")
        action, elapsed = fields[0], float(fields[1])
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed > external:
            raise RuntimeError("worker operational timeout")
        rss_guard()
        return action, elapsed


def play(row, candidate_player, candidate, opponent, candidate_build, opponent_build, smoke_ms=None,
         clock_policy=DEPLOYMENT_PREFIX_CLOCK, decision_guard=None):
    position = state(row["transcript"])
    prefix = "" if row["transcript"] in ("", "-") else row["transcript"]
    actors = [None, None]
    builds = [None, None]
    actors[candidate_player], actors[1-candidate_player] = candidate, opponent
    builds[candidate_player], builds[1-candidate_player] = candidate_build, opponent_build
    workers = []
    counts = prior_decision_counts(prefix, clock_policy)
    initial_counts = list(counts)
    decisions = []
    failure = None
    try:
        for build in builds:
            workers.append(StateWorker(build))
        for _ in range(320):
            if position.winner is not None:
                break
            if decision_guard is not None:
                decision_guard()
            mover = position.to_move
            first = counts[mover] == 0
            budget = smoke_ms or actors[mover]["clocks_ms"][0 if first else 1]
            counts[mover] += 1
            action, elapsed = workers[mover].choose(prefix, budget, 1000 if first else 200)
            rules.apply_complete_turn(position, mover, action)
            decisions.append({"player": mover, "action": action, "elapsed_ms": elapsed, "budget_ms": budget, "first": first})
            prefix += ("/" if prefix else "") + action
        if position.winner is None:
            raise RuntimeError("unfinished game")
    except (ValueError, RuntimeError, OSError, BrokenPipeError) as error:
        failure = str(error)
    finally:
        for worker in workers:
            worker.close()
    return {"candidate_player": candidate_player, "winner": position.winner,
            "candidate_won": position.winner == candidate_player if failure is None else None,
            "failure": failure, "decisions": decisions, "transcript": prefix,
            "root": row["state_sha256"], "cluster_id": row["cluster_id"], "opponent": row["opponent"],
            "clock_policy": clock_policy, "initial_own_decision_counts": initial_counts}


def bootstrap(rows, seed=20260916, repetitions=10000):
    clusters = collections.defaultdict(list)
    for row in rows:
        clusters[row["cluster_id"]].append(row["uplift"])
    keys = sorted(clusters)
    if not keys:
        raise ValueError("no independent clusters")
    rng = random.Random(seed)
    samples = []
    for _ in range(repetitions):
        values = [v for key in rng.choices(keys, k=len(keys)) for v in clusters[key]]
        samples.append(sum(values) / len(values))
    samples.sort()
    return {"clusters": len(keys), "repetitions": repetitions, "seed": seed,
            "lower_95": samples[int(.025 * (repetitions - 1))], "upper_95": samples[int(.975 * (repetitions - 1))]}


def assess(results, stage):
    failures = [g for r in results for arm in ("candidate", "control") for g in r[arm] if g["failure"]]
    if failures:
        return {"passed": False, "reason": "operational failure", "failures": len(failures)}
    paired = [{"cluster_id": r["root"]["cluster_id"], "opponent": r["root"]["opponent"],
               "uplift": (sum(g["candidate_won"] for g in r["candidate"]) - sum(g["candidate_won"] for g in r["control"])) / 2} for r in results]
    by_opponent = collections.defaultdict(list)
    for row in paired:
        by_opponent[row["opponent"]].append(row["uplift"])
    means = {name: sum(values) / len(values) for name, values in by_opponent.items()}
    mean = sum(r["uplift"] for r in paired) / len(paired)
    interval = bootstrap(paired)
    passes = stage != "smoke" and mean >= .03 and min(means.values()) >= -.05 and (stage != "confirmation" or interval["lower_95"] > 0)
    return {"passed": passes, "strength_eligible": stage != "smoke", "mean_uplift": mean,
            "opponent_uplift": means, "cluster_bootstrap": interval, "failures": 0}


def execution_order(index):
    return ("candidate", "control") if index % 2 == 0 else ("control", "candidate")


def validate_pair(result, row, candidate, control, opponent, smoke_ms,
                  clock_policy=DEPLOYMENT_PREFIX_CLOCK):
    if result["root"] != row:
        raise ValueError("cached pair changed root")
    for arm, actor in (("candidate", candidate), ("control", control)):
        if len(result[arm]) != 2:
            raise ValueError("pair omits physical color")
        for color, game in enumerate(result[arm]):
            if (game["candidate_player"] != color or game["cluster_id"] != row["cluster_id"] or
                    game["root"] != row["state_sha256"] or game["opponent"] != row["opponent"]):
                raise ValueError("cached pair identity mismatch")
            position = state(row["transcript"])
            counts = prior_decision_counts(row["transcript"], clock_policy)
            if game.get("clock_policy") != clock_policy or game.get("initial_own_decision_counts") != counts:
                raise ValueError("cached pair changed its prefix-derived clock policy")
            prefix = "" if row["transcript"] in ("", "-") else row["transcript"]
            for decision in game["decisions"]:
                player = position.to_move
                first = counts[player] == 0
                clocks = actor["clocks_ms"] if player == color else opponent["clocks_ms"]
                expected = smoke_ms or clocks[0 if first else 1]
                ms = decision["elapsed_ms"]
                if (decision["player"] != player or decision["first"] != first or decision["budget_ms"] != expected or
                        not math.isfinite(ms) or ms < 0 or ms > (1000 if first else 200)):
                    raise ValueError("cached pair timing/profile mismatch")
                rules.apply_complete_turn(position, player, decision["action"])
                prefix += ("/" if prefix else "") + decision["action"]
                counts[player] += 1
            if prefix != game["transcript"] or game["winner"] != position.winner:
                raise ValueError("cached pair replay mismatch")
            if game["failure"] is None and (position.winner is None or game["candidate_won"] != (position.winner == color)):
                raise ValueError("cached pair result mismatch")


def verify_candidate_reservation(bank, candidates):
    """Refuse a different candidate before compiling or claiming any game."""
    reservation = bank.get("candidate_source")
    if bank.get("generator_schema") == "papersoccer.top-three.development-bank-generator.v2" and reservation is None:
        raise ValueError("v2 bank lacks its candidate source reservation")
    if reservation is None:
        return
    if not isinstance(reservation, dict) or set(reservation) != {"path", "sha256"}:
        raise ValueError("invalid candidate source reservation")
    verify(reservation)
    if any(actor.get("source", {}).get("sha256") != reservation["sha256"] for actor in candidates):
        raise ValueError("confirmation bank is reserved for a different candidate source")


def run(root, candidate_path, control_path, bank_path, output, compiler, workers=1, smoke_ms=None):
    if not 1 <= workers <= 4:
        raise ValueError("worker limit is four")
    opponents = roster(root)
    candidate, control, bank = read(candidate_path), read(control_path), read(bank_path)
    verify_candidate_reservation(bank, [candidate])
    for actor in (candidate, control):
        descriptor(verify(actor["source"]), actor["family"], actor["clocks_ms"])
    if smoke_ms is not None and (bank["stage"] != "smoke" or not 1 <= smoke_ms <= 20):
        raise ValueError("reduced clocks are smoke-only")
    validate_bank(bank, opponents)
    output = Path(output).resolve()
    signature = compiler_signature(compiler)
    if (output / "plan.json").exists():
        verify_resume_builds(read(output / "plan.json"), signature)
    source_snapshots = {}
    for name, path in (("driver", Path(__file__)), ("worker", WORKER), ("rules", Path(rules.__file__))):
        raw = path.read_bytes()
        archived = output / "sources" / (campaign.digest(raw) + path.suffix)
        campaign.immutable(archived, raw)
        source_snapshots[name] = raw_record(archived)
    actors = {"candidate": candidate, "control": control, **opponents}
    builds = {name: build_worker(actor, output / "workers", signature["executable"]["path"])
              for name, actor in actors.items()}
    worker_builds = {name: verify_build_receipt(builds[name], actor, signature) for name, actor in actors.items()}
    bindings = {"schema": RUN_SCHEMA, "kind": "run", "candidate": raw_record(candidate_path),
                "control": raw_record(control_path), "bank": raw_record(bank_path), "opponents": opponents,
                "workers": workers, "rss_limit_bytes": RSS_LIMIT, "smoke_ms": smoke_ms,
                "compiler": signature, "worker_builds": worker_builds,
                "clock_policy": DEPLOYMENT_PREFIX_CLOCK,
                **source_snapshots}
    emit(output / "plan.json", bindings)
    def one(index_row):
        index, row = index_row
        if fingerprint(state(row["transcript"])) != row["state_sha256"]:
            raise ValueError("bank canonical root mismatch")
        path = output / "pairs" / f"{index:05}.json"
        if path.exists():
            result = read(path)
            validate_pair(result, row, candidate, control, opponents[row["opponent"]], smoke_ms)
            return result
        claim = output / "claims" / f"{index:05}.json"
        if claim.exists():
            raise ValueError("interrupted pair is spent; preserve output and freeze a fresh run")
        emit(claim, {"root": row, "plan": raw_record(output / "plan.json")})
        result = {"root": row}
        result["execution_order"] = list(execution_order(index))
        for arm in execution_order(index):
            actor = candidate if arm == "candidate" else control
            result[arm] = [play(row, color, actor, opponents[row["opponent"]], builds[arm], builds[row["opponent"]], smoke_ms) for color in (0, 1)]
        validate_pair(result, row, candidate, control, opponents[row["opponent"]], smoke_ms)
        emit(path, result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(one, enumerate(bank["rows"])))
    result = {"schema": SCHEMA, "kind": "result", "plan": raw_record(output / "plan.json"),
              "assessment": assess(results, bank["stage"]), "pairs": len(results),
              "pair_artifacts": [raw_record(output / "pairs" / f"{i:05}.json") for i in range(len(results))]}
    emit(output / "result.json", result)
    return result


def many_execution_order(index, names):
    arms = ("control", *sorted(names))
    offset = index % len(arms)
    return arms[offset:] + arms[:offset]


def paired_view(result, name):
    # Deliberately refer to the one stored control list, rather than creating
    # another control observation or running a new control game.
    return {"root": result["root"], "candidate": result["arms"][name], "control": result["arms"]["control"]}


def validate_many_root(result, row, index, candidates, control, opponent, smoke_ms, plan_identity):
    expected_names = {"control", *candidates}
    if (result.get("schema") != MANY_ROOT_SCHEMA or result.get("plan") != plan_identity or
            set(result.get("arms", {})) != expected_names or
            result.get("execution_order") != list(many_execution_order(index, candidates))):
        raise ValueError("shared-control root arm/order/plan identity changed")
    for name, candidate in candidates.items():
        validate_pair(paired_view(result, name), row, candidate, control, opponent, smoke_ms)


def assess_many(results, stage, names):
    names = tuple(sorted(names))
    if not results or not 1 <= len(names) <= 2 or len(set(names)) != len(names):
        raise ValueError("shared-control assessment requires one or two candidates")
    expected = {"control", *names}
    if any(set(r["arms"]) != expected for r in results):
        raise ValueError("shared-control assessment has a missing or extra arm")
    if len({r["root"]["state_sha256"] for r in results}) != len(results):
        raise ValueError("shared-control root is duplicated")
    if any(len(games) != 2 for r in results for games in r["arms"].values()):
        raise ValueError("shared-control arm omits a color")
    comparisons = {name: assess([paired_view(row, name) for row in results], stage) for name in names}
    independent = len({r["root"]["cluster_id"] for r in results})
    for value in comparisons.values():
        value["independent_root_clusters"] = independent
        value["candidate_games"] = 2 * len(results)
        value["shared_control_games"] = 2 * len(results)
    return {"comparisons": comparisons, "unique_games": 2 * len(results) * (len(names) + 1),
            "unique_control_games": 2 * len(results), "root_clusters": independent,
            "unique_operational_failures": sum(bool(g["failure"]) for r in results for arm in r["arms"].values() for g in arm),
            "comparisons_share_controls": len(names) > 1,
            "independence_note": "Each comparison resamples the same root/game clusters; shared controls are stored once and comparisons are correlated. No pooled independent-sample claim."}


def run_many(root, candidate_paths, control_path, bank_path, output, compiler, workers=1, smoke_ms=None):
    entries = list(candidate_paths.items()) if isinstance(candidate_paths, dict) else list(candidate_paths)
    if not 1 <= len(entries) <= 2 or not 1 <= workers <= 4:
        raise ValueError("run-many allows at most two candidates and four workers")
    names = [entry[0] for entry in entries]
    if (len(set(names)) != len(names) or "control" in names or
            any(not isinstance(n, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,47}", n) is None for n in names)):
        raise ValueError("candidate names must be unique safe labels, excluding control")
    paths = {name: Path(path) for name, path in sorted(entries)}
    opponents = roster(root)
    loaded = {name: load_snapshot(path) for name, path in paths.items()}
    candidates = {name: value[0] for name, value in loaded.items()}
    descriptor_inputs = {name: value[1] for name, value in loaded.items()}
    control, control_identity = load_snapshot(control_path)
    bank, bank_identity = load_snapshot(bank_path)
    verify_candidate_reservation(bank, candidates.values())
    for actor in (*candidates.values(), control):
        descriptor(verify(actor["source"]), actor["family"], actor["clocks_ms"])
    if len({actor["source"]["sha256"] for actor in candidates.values()}) != len(candidates):
        raise ValueError("identical candidate sources must not be counted as two candidates")
    if smoke_ms is not None and (bank["stage"] != "smoke" or not 1 <= smoke_ms <= 20):
        raise ValueError("reduced clocks are smoke-only")
    validate_bank(bank, opponents)
    output = Path(output).resolve()
    signature = compiler_signature(compiler)
    if (output / "plan.json").exists():
        previous = read(output / "plan.json")
        verify_resume_builds(previous, signature, expected_schema=MANY_RUN_SCHEMA)
        if (previous.get("candidates") != descriptor_inputs or previous.get("control") != control_identity
                or previous.get("bank") != bank_identity or previous.get("opponents") != opponents):
            raise ValueError("shared-control candidate/control/bank identity changed on resume")
    sources = {}
    for name, path in (("driver", Path(__file__)), ("worker", WORKER), ("rules", Path(rules.__file__))):
        raw = path.read_bytes()
        archived = output / "sources" / (campaign.digest(raw) + path.suffix)
        campaign.immutable(archived, raw)
        sources[name] = raw_record(archived)
    actors = {"control": control, **{"candidate:" + name: actor for name, actor in candidates.items()},
              **{"opponent:" + name: actor for name, actor in opponents.items()}}
    builds = {name: build_worker(actor, output / "workers", signature["executable"]["path"])
              for name, actor in actors.items()}
    worker_builds = {name: verify_build_receipt(builds[name], actor, signature) for name, actor in actors.items()}
    for identity in (*descriptor_inputs.values(), control_identity, bank_identity):
        verify(identity)
    bindings = {"schema": MANY_RUN_SCHEMA, "candidates": descriptor_inputs, "candidate_order": list(candidates),
                "control": control_identity, "bank": bank_identity, "opponents": opponents,
                "workers": workers, "rss_limit_bytes": RSS_LIMIT, "smoke_ms": smoke_ms,
                "compiler": signature, "worker_builds": worker_builds, "clock_policy": DEPLOYMENT_PREFIX_CLOCK,
                "arm_order_policy": "cyclic rotation of control then sorted candidate names by root index",
                "control_policy": "one two-color control arm per root shared by all comparisons", **sources}
    emit(output / "plan.json", bindings)
    plan_identity = raw_record(output / "plan.json")
    def one(index_row):
        index, row = index_row
        path = output / "roots" / f"{index:05}.json"
        if path.exists():
            result = read(path)
            validate_many_root(result, row, index, candidates, control, opponents[row["opponent"]], smoke_ms, plan_identity)
            return result
        claim = output / "claims" / f"{index:05}.json"
        if claim.exists():
            raise ValueError("interrupted shared-control root is preserved; no silent rerun")
        emit(claim, {"root": row, "plan": plan_identity, "execution_order": list(many_execution_order(index, candidates))})
        result = {"schema": MANY_ROOT_SCHEMA, "plan": plan_identity, "root": row,
                  "execution_order": list(many_execution_order(index, candidates)), "arms": {}}
        for name in many_execution_order(index, candidates):
            actor = control if name == "control" else candidates[name]
            build_name = "control" if name == "control" else "candidate:" + name
            result["arms"][name] = [play(row, color, actor, opponents[row["opponent"]],
                builds[build_name], builds["opponent:" + row["opponent"]], smoke_ms) for color in (0, 1)]
        validate_many_root(result, row, index, candidates, control, opponents[row["opponent"]], smoke_ms, plan_identity)
        emit(path, result)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(one, enumerate(bank["rows"])))
    result = {"schema": MANY_RESULT_SCHEMA, "plan": plan_identity,
              "assessment": assess_many(results, bank["stage"], candidates), "roots": len(results),
              "root_artifacts": [raw_record(output / "roots" / f"{i:05}.json") for i in range(len(results))]}
    emit(output / "result.json", result)
    return result


def clean_initial_game(game):
    if game.get("schema") != "papersoccer.codingame-match.v1":
        return False
    outcome = game.get("outcome", {})
    actions = game.get("actions", [])
    if (not actions or outcome.get("forfeit") is not None or outcome.get("reason") not in ("goal", "blocked_mover")):
        return False
    position = rules.ReplayState()
    try:
        for action in actions:
            if action.get("accepted") is not True or action.get("failureClassification") is not None:
                return False
            if action["durationMicros"] > action["deadlineMillis"] * 1000:
                return False
            rules.apply_complete_turn(position, action["player"], action["action"])
        winner = game["participants"]["playerOne" if position.winner == 0 else "playerTwo"]["id"]
        return position.winner is not None and winner == outcome["winnerId"]
    except (KeyError, TypeError, ValueError):
        return False


def initial(referee, first, second, output):
    """True fresh-process empty-board games in both colors, not independent roots."""
    referee, first, second = (Path(p).resolve() for p in (referee, first, second))
    reference = raw_record(referee)
    binaries = [raw_record(first), raw_record(second)]
    games = []
    for color in (0, 1):
        args = [str(referee), "--player-one", str((first, second)[color]), "--player-two", str((second, first)[color]),
                "--player-one-id", ("candidate", "opponent")[color], "--player-two-id", ("opponent", "candidate")[color]]
        result = subprocess.run(args, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(result.stderr)
        games.append(json.loads(result.stdout))
    value = {"schema": SCHEMA, "kind": "initial-protocol", "referee": reference,
             "binaries": binaries, "independent_roots": 1, "games": games,
             "zero_operational_failures": all(clean_initial_game(g) for g in games)}
    emit(output, value)
    return value


def training_trajectory(envelope):
    """Validate one fresh training producer envelope, never an arena replay.

    Producer freezes source_kind/seed/split/root_group_id before play. The seed
    identifies procedural root generation; exact archived engines keep their
    own internal randomization. This API does not allocate games or teach labels.
    """
    root, game = envelope["root"], envelope["game"]
    if root.get("source_kind") != "fresh-generated" or game["failure"] is not None:
        raise ValueError("only clean, fresh-generated training games are exportable")
    if envelope["split"] not in ("train", "validation") or type(envelope["seed"]) is not int:
        raise ValueError("training split/seed missing")
    if not isinstance(root["root_group_id"], str) or not root["root_group_id"]:
        raise ValueError("training root group missing")
    before = state(root["transcript"])
    if before.winner is not None:
        raise ValueError("training root already terminal")
    canonical = fingerprint(before)
    prefix = "" if root["transcript"] in ("", "-") else root["transcript"]
    if prefix and not game["transcript"].startswith(prefix + "/"):
        raise ValueError("training transcript does not extend root")
    after = state(game["transcript"])
    if after.winner is None or after.winner != game["winner"]:
        raise ValueError("training transcript winner mismatch")
    candidate = envelope["candidate"]
    opponent = envelope["opponent"]
    verify(candidate["source"])
    verify(opponent["source"])
    # Validate the actual decision clocks; reduced-clock learning generation
    # must declare reduced clocks in its descriptors, never inherit a strength
    # profile while silently calling play(..., smoke_ms=...).
    actors = {"candidate": candidate, "opponent": opponent}
    played = state(root["transcript"])
    clock_policy = game.get("clock_policy", FRESH_ROOT_CLOCK)
    counts = prior_decision_counts(root["transcript"], clock_policy)
    rebuilt = prefix
    for decision in game["decisions"]:
        player = played.to_move
        if decision["player"] != player or decision["first"] != (counts[player] == 0):
            raise ValueError("training decision mover/clock phase mismatch")
        actor = "candidate" if player == game["candidate_player"] else "opponent"
        if decision["budget_ms"] != actors[actor]["clocks_ms"][0 if decision["first"] else 1]:
            raise ValueError("training decision clock differs from declared profile")
        ms = decision["elapsed_ms"]
        if not math.isfinite(ms) or ms < 0 or ms > (1000 if decision["first"] else 200):
            raise ValueError("training decision operational failure")
        rules.apply_complete_turn(played, player, decision["action"])
        rebuilt += ("/" if rebuilt else "") + decision["action"]
        counts[player] += 1
    if rebuilt != game["transcript"] or played.winner != after.winner:
        raise ValueError("training decisions do not reproduce transcript")
    return {"schema": "papersoccer.top-three.training-game.v1", "game_id": envelope["game_id"],
            "root_group_id": root["root_group_id"], "root_state_sha256": canonical,
            "seed": envelope["seed"], "seed_role": "procedural-root-generation; archived engine settings unchanged",
            "opponent": envelope["opponent_name"], "source_sha256": {
                "candidate": candidate["source"]["sha256"], "opponent": opponent["source"]["sha256"]},
            "transcript": game["transcript"], "winner": after.winner,
            "split": envelope["split"], "training_eligible": True, "source_kind": "fresh-generated",
            "root_transcript": prefix, "root_drawn_edges": len(before.used_segments), "rule_terminal": True,
            "candidate_player": game["candidate_player"], "search_profiles": {
                "candidate": candidate["clocks_ms"], "opponent": opponent["clocks_ms"]},
            "clock_policy": clock_policy}


def export_training(input_path, output):
    rows = [training_trajectory(json.loads(line)) for line in Path(input_path).read_text().splitlines()]
    ids, groups, canonical_groups = set(), {}, {}
    for row in rows:
        if row["game_id"] in ids:
            raise ValueError("duplicate training game ID")
        ids.add(row["game_id"])
        group, split, key = row["root_group_id"], row["split"], row["root_state_sha256"]
        if group in groups and groups[group] != split:
            raise ValueError("root group crosses training/validation")
        if key in canonical_groups and canonical_groups[key] != group:
            raise ValueError("canonical/symmetric root relabelled as independent group")
        groups[group] = split
        canonical_groups[key] = group
    content = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    campaign.immutable(Path(output), content)
    return {"schema": SCHEMA, "kind": "training-export", "output": raw_record(output),
            "input": raw_record(input_path), "games": len(rows),
            "split_games": dict(collections.Counter(r["split"] for r in rows)),
            "split_root_groups": dict(collections.Counter(groups.values())),
            "early_group_coverage": "must be measured by the label/training consumer; game count alone is insufficient"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=campaign.DEFAULT)
    commands = parser.add_subparsers(dest="command", required=True)
    actor = commands.add_parser("descriptor")
    actor.add_argument("--source", type=Path, required=True)
    actor.add_argument("--family", choices=FAMILIES, required=True)
    actor.add_argument("--clocks-ms", type=int, nargs=2, required=True)
    actor.add_argument("--output", type=Path, required=True)
    bank = commands.add_parser("freeze-bank")
    bank.add_argument("--input", type=Path, required=True)
    bank.add_argument("--output", type=Path, required=True)
    bank.add_argument("--stage", choices=("smoke", "screen", "confirmation"), required=True)
    bank.add_argument("--exclude-bank", type=Path, action="append", default=[])
    runner = commands.add_parser("run")
    for name in ("candidate", "control", "bank", "output"):
        runner.add_argument("--" + name, type=Path, required=True)
    runner.add_argument("--compiler", default="/usr/bin/clang++")
    runner.add_argument("--workers", type=int, default=1)
    runner.add_argument("--smoke-ms", type=int)
    many = commands.add_parser("run-many")
    many.add_argument("--candidate", nargs=2, action="append", metavar=("NAME", "DESCRIPTOR"), required=True)
    for name in ("control", "bank", "output"):
        many.add_argument("--" + name, type=Path, required=True)
    many.add_argument("--compiler", default="/usr/bin/clang++")
    many.add_argument("--workers", type=int, default=1)
    many.add_argument("--smoke-ms", type=int)
    first = commands.add_parser("initial")
    for name in ("referee", "candidate", "opponent", "output"):
        first.add_argument("--" + name, type=Path, required=True)
    training = commands.add_parser("export-training")
    training.add_argument("--input", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "descriptor":
        result = descriptor(args.source, args.family, args.clocks_ms)
        emit(args.output, result)
    elif args.command == "freeze-bank":
        result = freeze_bank(args.root, args.input, args.output, args.stage, args.exclude_bank)
    elif args.command == "run":
        result = run(args.root, args.candidate, args.control, args.bank, args.output, args.compiler, args.workers, args.smoke_ms)
    elif args.command == "run-many":
        result = run_many(args.root, args.candidate, args.control, args.bank, args.output, args.compiler, args.workers, args.smoke_ms)
    elif args.command == "initial":
        result = initial(args.referee, args.candidate, args.opponent, args.output)
    else:
        result = export_training(args.input, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
