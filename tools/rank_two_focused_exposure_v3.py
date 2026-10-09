"""Increment explicit new exposures into a copied, receipt-bound ancestry index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import sys

from tools import jacek_replay_corpus as teacher
from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_prior_v1 as prior_validator
from tools import top_three_development_banks as proposals
from tools import top_three_played_exclusions_v3 as census

SCHEMA = campaign.SCHEMA + ".incremental-exposure"
FOLLOW = {"input", "inputs", "result", "report", "cases", "bank", "snapshot", "raw",
          "proposal_receipts", "generation_plan", "game_receipts", "parent_receipts", "execution", "output",
          "all64exposures", "all256exposures", "all64game_receipts", "all256game_receipts"}
EXTERNAL_PRODUCERS = {Path(path).resolve() for path in (
    "/usr/bin/clang++", "/usr/bin/time",
    sys.executable)}


def verify(reference):
    path = Path(reference["path"]).resolve()
    if not path.is_relative_to(campaign.ROOT) and path not in EXTERNAL_PRODUCERS:
        raise ValueError("exposure input escapes the approved worktree")
    bound = campaign.record(path)
    if bound["sha256"] != reference["sha256"] or ("bytes" in reference and path.stat().st_size != reference["bytes"]):
        raise ValueError("changed exposure input")
    return path


def proposal_trace(seed, depth):
    """Replay only this new receipt's exact bounded walk, including its tail."""
    if type(seed) is not int or type(depth) is not int or not 0 <= depth <= 316:
        raise ValueError("bounded proposal seed/depth required")
    rng, state = random.Random(seed), census.e.rules.ReplayState()
    actions, action, error = [], "", None
    for _ in range(depth):
        mover = state.to_move
        x, y = state.ball
        options = [direction for direction, (dx, dy) in enumerate(census.e.rules.DIRECTION_DELTAS)
                   if census.e.rules._legal_destination(state, (x + dx, y + dy))]
        if not options:
            error = "no-legal-edge"
            break
        direction = options[rng.randrange(len(options))]
        census.e.rules.apply_primitive(state, direction)
        action += str(direction)
        if state.winner is not None:
            error = "terminal"
            break
        if state.to_move != mover:
            actions.append(action)
            action = ""
    if error is None and action:
        error = "incomplete-turn-boundary"
    original, transcript, original_error = proposals.propose(seed, depth)
    if original_error != error or (error is None and (
            transcript != "/".join(actions) or census.e.fingerprint(original) != census.e.fingerprint(state))):
        raise ValueError("proposal replay changed the frozen algorithm")
    return actions + ([action] if action else []), error


class Normalizer:
    def __init__(self, stream):
        self.stream, self.rows, self.inputs, self.parsed = stream, 0, {}, set()
        self.recipe_only, self.seen_rows = [], set()

    def emit(self, turns, reference, locator, kind="focused-exclusion-only", group=None):
        if isinstance(turns, str):
            turns = [] if turns in ("", "-") else turns.split("/")
        if not isinstance(turns, list) or any(not isinstance(row, (str, dict)) for row in turns):
            raise ValueError("unrecognized exposed transcript")
        if any(isinstance(row, str) and any(ch not in "01234567" for ch in row) for row in turns):
            raise ValueError("exposure transcript is not primitive directions")
        row = dict(mode="trace", turns=turns, input=reference, locator=locator, kind=kind,
                   group=str(group) if group is not None else "recipe:" + reference["sha256"], training_eligible=False)
        raw = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if digest not in self.seen_rows:
            self.seen_rows.add(digest)
            self.stream.write(raw)
            self.rows += 1

    def bind(self, reference, parse=False):
        path = verify(reference)
        simple = campaign.record(path)
        previous = self.inputs.get(str(path))
        if previous is not None and previous != simple:
            raise ValueError("one input has conflicting exposure identities")
        self.inputs[str(path)] = simple
        if len(self.inputs) > 200000:
            raise ValueError("explicit exposure input count exceeded")
        if not parse or str(path) in self.parsed:
            return
        self.parsed.add(str(path))
        if not path.is_relative_to(campaign.ROOT):
            raise ValueError("external producer may be bound but never parsed as campaign data")
        if path.suffix == ".jsonl":
            with path.open() as stream:
                for index, line in enumerate(stream):
                    if line.strip():
                        self.walk(json.loads(line), simple, f"/{index}")
        elif path.suffix == ".json":
            if path.stat().st_size > 256 * 1024**2:
                raise ValueError("oversized explicit exposure document")
            self.walk(campaign.read(path), simple, "")
        else:
            self.recipe_only.append(simple)

    def walk(self, value, reference, pointer, key=None):
        if isinstance(value, list):
            for index, row in enumerate(value):
                self.walk(row, reference, pointer + "/" + str(index), key)
            return
        if not isinstance(value, dict):
            return
        if "path" in value and "sha256" in value:
            self.bind(value, key in FOLLOW)
            return
        # Native successors are mover-relative single turns, not empty-board
        # transcripts. Convert them with the accepted validator's exact rule.
        native_group = value.get("group")
        if isinstance(native_group, dict) and "source_binding" in native_group and "successors" in native_group:
            validated = teacher.validate_complete_turn_action_group(value)
            group = validated["group"]
            prefix = [row["action"] for row in group["source_binding"]["prefix"]]
            self.emit(prefix, reference, pointer + "/group/source_binding/prefix", "focused-teacher-parent",
                      group["source_binding"]["root_group_id"])
            for index, row in enumerate(group["successors"]):
                action = teacher._canonical_transcript_for_physical(row["transcript"], group["parent_mover"])
                self.emit(prefix + [action], reference, pointer + f"/group/successors/{index}/transcript",
                          "focused-teacher-successor", group["source_binding"]["root_group_id"])
            return
        if "proposal_seed" in value and "drawn_edges" in value and "status" in value:
            turns, error = proposal_trace(int(value["proposal_seed"]), value["drawn_edges"])
            if error is not None and value["status"] != error:
                raise ValueError("proposal receipt disagrees with its frozen recipe")
            self.emit(turns, reference, pointer, "focused-reserved-proposal-and-partial", value.get("slot"))
        group = value.get("root_group_id", value.get("cluster_id", value.get("game_id")))
        for name in ("transcript", "prefix", "root_transcript"):
            text = value.get(name)
            if isinstance(text, str):
                self.emit(text, reference, pointer + "/" + name, group=group)
        if isinstance(value.get("turns"), list):
            self.emit(value["turns"], reference, pointer + "/turns", group=group)
        prefix = value.get("prefix")
        if isinstance(prefix, str):
            partial = value.get("partial", "")
            for field in ("directions", "legal_directions"):
                if field in value:
                    for direction in value[field]:
                        self.emit(([] if prefix in ("", "-") else prefix.split("/")) + [partial + str(direction)],
                                  reference, pointer + "/" + field + "/" + str(direction), "focused-primitive-proposal")
            for index, alternative in enumerate(value.get("alternatives", [])):
                if isinstance(alternative, dict) and type(alternative.get("direction")) is int:
                    self.emit(([] if prefix in ("", "-") else prefix.split("/"))
                              + [partial + str(alternative["direction"])], reference,
                              pointer + f"/alternatives/{index}/direction", "focused-primitive-proposal")
            if isinstance(value.get("action"), str) and value["action"]:
                self.emit(([] if prefix in ("", "-") else prefix.split("/")) + [value["action"]],
                          reference, pointer + "/action", "focused-returned-action")
            for index, choice in enumerate(value.get("choices", [])):
                if isinstance(choice, dict):
                    self.walk({**choice, "prefix": prefix}, reference, pointer + f"/choices/{index}")
        for index, text in enumerate(value.get("canonical_prefixes", [])):
            self.emit(text, reference, pointer + f"/canonical_prefixes/{index}", "focused-unit-canonical-fixture")
        # A teacher exposure's plan contains the retained parent inputs. Ordinary
        # plan/prior/inventory/database references are only bound, never crawled.
        for name, child in value.items():
            if name == "choices" and isinstance(prefix, str):
                continue
            follow_name = "inputs" if name == "plan" and str(value.get("schema", "")).endswith("teacher-exposure") else name
            self.walk(child, reference, pointer + "/" + name, follow_name)


def prepare(base, seed_ready, exposures, output):
    base, output = Path(base).resolve(), Path(output).resolve()
    current = campaign.read(base / "CURRENT.json")
    if current["ownership_status"] != "active":
        raise PermissionError("campaign owner inactive")
    seed = campaign.read(seed_ready)
    if seed.get("passed") is not True:
        raise ValueError("completed ancestry seed required")
    prior = campaign.read(verify(seed["prior"]))
    with prior_validator.prior_census(prior):
        pass
    output.mkdir(parents=True, exist_ok=True)
    cutoff = campaign.immutable(output / "CUTOFF.json", dict(cutoff_utc=campaign.now().isoformat(),
        excluded_current_live_details_directory=str(output / "protected-inputs-not-permitted"),
        historical_protected_reads=False, purpose="explicit new exposure delta; no new games or strength analysis"))
    scope = campaign.immutable(output / "scope.json", dict(schema=census.v2.SCOPE_SCHEMA,
        campaign_root=str(campaign.ROOT / "results"), cutoff=cutoff, entries=[],
        inherit=[dict(kind="exposure", input=prior["inputs"]["inventory"])], historical_protected_reads=False))
    records = output / "records.jsonl"
    with records.open("x") as stream:
        normal = Normalizer(stream)
        for reference in exposures:
            normal.bind(reference, parse=True)
    closure = campaign.immutable(output / "records.closure.json", dict(
        schema="papersoccer.top-three.normalized-exposure-closure.v1", scope=scope, cutoff=cutoff,
        records=campaign.record(records), rows=normal.rows,
        inputs=sorted(normal.inputs.values(), key=lambda item: item["path"]), skipped_or_unreturned=[],
        bound_recipe_only=normal.recipe_only, all_payload_inputs_hash_verified_before_census=True,
        producer=campaign.record(__file__)))
    plan = dict(schema=SCHEMA, seed_ready=campaign.record(seed_ready), seed_prior=seed["prior"],
                seed_inputs=prior["inputs"], exposures=exposures, closure=closure, scope=scope,
                records=campaign.record(records), rows=normal.rows, producer=campaign.record(__file__),
                producers=census.producers(), proposal_algorithm=campaign.record(proposals.__file__),
                teacher_validator=campaign.record(teacher.__file__), prior_validator=campaign.record(prior_validator.__file__),
                owner_thread_id=current["owner_thread_id"], activation=current["focused_activation"],
                campaign_current=str(base / "CURRENT.json"),
                cpu_seconds=1200, maximum_rss_bytes=4 * 1024**3, historical_protected_reads=False,
                games_run=0, historical_ancestry_rescan=False)
    campaign.immutable(output / "PLAN.json", plan)
    return plan


def extend(plan_path):
    plan, output = campaign.read(plan_path), Path(plan_path).resolve().parent
    if plan["schema"] != SCHEMA or plan["producer"] != campaign.record(__file__):
        raise ValueError("incremental exposure producer differs")
    def owner_check():
        current = campaign.read(plan["campaign_current"])
        if (current["owner_thread_id"] != plan["owner_thread_id"] or current["ownership_status"] != "active"
                or current["focused_activation"] != plan["activation"]
                or current.get("human_pause", {}).get("status") == "paused"):
            raise PermissionError("ownership/pause/activation changed; ancestry claim retained")
    owner_check()
    for ref in [plan["closure"], plan["scope"], plan["records"], plan["proposal_algorithm"],
                plan["teacher_validator"], plan["prior_validator"], *plan["producers"].values()]:
        verify(ref)
    closure, scope = campaign.read(plan["closure"]["path"]), campaign.read(plan["scope"]["path"])
    for ref in closure["inputs"]:
        verify(ref)
    prior = campaign.read(verify(plan["seed_prior"]))
    if prior["inputs"] != plan["seed_inputs"]:
        raise ValueError("ancestry seed chain changed")
    with prior_validator.prior_census(prior):
        pass
    if (output / "claim.json").exists():
        raise RuntimeError("unknown spent census claim retained; no automatic restart")
    campaign.immutable(output / "claim.json", dict(plan=campaign.record(plan_path)))
    directory = output / "census"
    directory.mkdir(exist_ok=True)
    binding = dict(schema=census.SCHEMA, scope=plan["scope"], closure=plan["closure"], records=plan["records"],
        sources=plan["producers"], inherited_inputs=[prior["inputs"]["inventory"]],
        indexed_seed=dict(database=prior["inputs"]["database"], receipt=prior["inputs"]["receipt"],
                          plan=campaign.record(plan_path)), extension_producer=plan["producer"])
    campaign.immutable(directory / "binding.json", binding)
    database = directory / "checkpoint.sqlite3"
    source = sqlite3.connect(Path(prior["inputs"]["database"]["path"]).as_uri() + "?mode=ro", uri=True)
    db = sqlite3.connect(database)
    budget = census.Budget(plan["cpu_seconds"], plan["maximum_rss_bytes"])
    try:
        for connection in (source, db):
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("PRAGMA mmap_size=0")
        source.backup(db, pages=256)
        with db:
            db.execute("DELETE FROM completed")
            db.execute("DELETE FROM metadata")
            bootstrap = dict(completed=0, historical_counts_unconfirmed={},
                inherited_inventories=[dict(kind="exposure", input=prior["inputs"]["inventory"])],
                indexed_seed_receipt=prior["inputs"]["receipt"])
            db.execute("INSERT INTO metadata VALUES (?,?)", ("bootstrap", json.dumps(bootstrap, sort_keys=True)))
            db.execute("INSERT INTO metadata VALUES (?,?)", ("binding", json.dumps(binding, sort_keys=True)))
        with verify(plan["records"]).open() as stream:
            for number, line in enumerate(stream, 1):
                if number % 50 == 1:
                    owner_check()
                row = json.loads(line)
                item = census.RowCensus(scope["campaign_root"], db, budget)
                item.trace(row["turns"], row["input"], row["locator"], row["kind"], row["group"])
                with db:
                    for category in ("states", "features", "groups"):
                        census.union(db, category, getattr(item, category))
                    # Retain the new teacher ancestry in the inherited namespace;
                    # all these entries remain exclusion-only, never targets.
                    if row["kind"].startswith("focused-teacher"):
                        census.union(db, "teacher:focused_observed_states", item.states)
                        census.union(db, "teacher:focused_observed_features", item.features)
                    db.execute("INSERT INTO completed VALUES (?,?,?,?)", (number,
                        hashlib.sha256(line.encode()).hexdigest(), json.dumps(dict(item.counts)), json.dumps(item.ledger)))
                budget.guard()
        if census.checkpoint(db) != plan["rows"]:
            raise ValueError("incremental ancestry row frontier incomplete")
        result = census.export_inventory(db, directory / "inventory-001.json", binding, scope, closure, "complete", None, budget)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        result.update(schema=census.SCHEMA, checkpoint=str(database.resolve()),
                      historical_protected_reads=False, historical_ancestry_rescan=False,
                      games_run=0, workers=1, rss_cap_bytes=plan["maximum_rss_bytes"], cpu_cap_seconds=plan["cpu_seconds"])
        campaign.immutable(directory / "receipt-001.json", result)
    finally:
        db.close(); source.close()
    frozen = dict(schema="papersoccer.rank-two.live-prior-exposure.v1", historical_protected_reads=False,
        packet_id="focused-network-exposure-delta", extension_plan=campaign.record(plan_path),
        inputs=dict(inventory=result["inventory"], database=campaign.record(database),
                    receipt=campaign.record(directory / "receipt-001.json"), binding=campaign.record(directory / "binding.json"),
                    closure=plan["closure"], scope=plan["scope"]))
    with prior_validator.prior_census(frozen):
        pass
    campaign.immutable(output / "bank-prior.json", frozen)
    ready = dict(passed=True, plan=campaign.record(plan_path), prior=campaign.record(output / "bank-prior.json"),
                 inventory=result["inventory"], receipt=campaign.record(directory / "receipt-001.json"),
                 rows=plan["rows"], exposures=plan["exposures"], games_run=0,
                 historical_protected_reads=False, historical_ancestry_rescan=False)
    campaign.immutable(output / "ready.json", ready)
    return ready


def run_bundle(input_path, output):
    inputs = campaign.read(input_path)
    if inputs["producer"] != campaign.record(__file__):
        raise ValueError("bound exposure preparation producer changed")
    current = campaign.read(Path(inputs["base"]) / "CURRENT.json")
    if current["owner_thread_id"] != inputs["owner_thread_id"] or current["focused_activation"] != inputs["activation"]:
        raise PermissionError("bound exposure preparation owner changed")
    verify(inputs["seed_ready"])
    for reference in inputs["exposures"]:
        verify(reference)
    prepare(inputs["base"], inputs["seed_ready"]["path"], inputs["exposures"], output)
    return extend(Path(output) / "PLAN.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", type=Path)
    group.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.inputs and args.output is None:
        parser.error("--inputs requires --output")
    print(json.dumps(extend(args.plan) if args.plan else run_bundle(args.inputs, args.output)))


if __name__ == "__main__":
    main()
