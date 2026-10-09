"""Bounded continuation of committed exposure rows without replaying the frontier."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_prior_v1 as validator
from tools import rank_two_focused_work_v1 as work
from tools import top_three_played_exclusions_v3 as census


class CachedRows(census.RowCensus):
    """Caches are valid only after the preceding row transaction commits."""
    def __init__(self, root, db, budget, physical, traces, teacher_observation=False):
        super().__init__(root, db, budget)
        self.db = db
        self.physical_seen = {} if teacher_observation else physical
        self.cache = {} if teacher_observation else traces
        self.teacher_observation = teacher_observation
        self.indexed_reuses = 0

    def observe(self, state):
        self.observations += 1
        if self.observations % 256 == 0:
            self.guard()
        physical = (state.ball, state.to_move, state.winner,
                    sum(census.v2.EDGE_BITS[edge] for edge in state.used_segments))
        known = self.physical_seen.get(physical)
        if known is not None:
            return known
        canonical = census.e.fingerprint(state)
        exists = self.db.execute("SELECT 1 FROM keys WHERE category='states' AND value=?", (canonical,)).fetchone()
        if exists is not None and not self.teacher_observation:
            # The inherited index already contains this state's canonical
            # feature. Re-encoding it would rescan prior ancestry needlessly.
            self.indexed_reuses += 1
        else:
            feature = census.banks.feature_key(state)
            self.states.add(canonical)
            self.features.add(feature)
            self.feature_by_state[canonical] = feature
        self.physical_seen[physical] = canonical
        return canonical


def resume(plan_path):
    plan = campaign.read(plan_path)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("continuation producer changed")
    work.checked(plan, "archive", launch=True)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "CLAIM.json").exists():
        raise RuntimeError("unknown continuation claim retained; explicit classification required")
    original = campaign.read(campaign.verify(plan["normalized_plan"]))
    for name in ("records", "scope", "closure"):
        campaign.verify(original[name])
    for ref in original["producers"].values():
        campaign.verify(ref)
    campaign.verify(plan["parent_binding"])
    campaign.verify(plan["parent_database"])
    if plan.get("parent_wal"):
        campaign.verify(plan["parent_wal"])
    current = campaign.read(plan["campaign_current"])
    if (current["owner_thread_id"] != plan["owner_thread_id"] or current["ownership_status"] != "active"
            or current["focused_activation"] != plan["activation"]):
        raise PermissionError("continuation ownership/activation changed")
    campaign.immutable(output / "CLAIM.json", dict(plan=campaign.record(plan_path),
                                                frontier=plan["committed_rows"], maximum_new_rows=plan["maximum_new_rows"]))
    # Copy the data/WAL pair before opening SQLite. The stopped parent's bytes
    # remain untouched even if SQLite creates a new shared-memory index.
    snapshot = output / "parent-snapshot.sqlite3"
    shutil.copyfile(campaign.verify(plan["parent_database"]), snapshot)
    if plan.get("parent_wal"):
        shutil.copyfile(campaign.verify(plan["parent_wal"]), Path(str(snapshot) + "-wal"))
    directory = output / "census"
    directory.mkdir(exist_ok=True)
    database = directory / "checkpoint.sqlite3"
    source = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
    db = sqlite3.connect(database)
    budget = census.Budget(plan["cpu_seconds"], plan["maximum_rss_bytes"])
    binding = campaign.read(plan["parent_binding"]["path"])
    binding = {**binding, "continuation_producer": plan["producer"],
               "continuation_plan": campaign.record(plan_path), "committed_parent_rows": plan["committed_rows"]}
    scope = campaign.read(original["scope"]["path"])
    closure = campaign.read(original["closure"]["path"])
    physical, traces = {}, {}
    committed, reused = 0, 0
    try:
        for connection in (source, db):
            connection.execute("PRAGMA cache_size=-32768")
            connection.execute("PRAGMA mmap_size=0")
        source.backup(db, pages=256)
        parent_binding = json.loads(db.execute("SELECT value FROM metadata WHERE name='binding'").fetchone()[0])
        if parent_binding != campaign.read(plan["parent_binding"]["path"]):
            raise ValueError("parent data/WAL binding differs")
        frontier = census.checkpoint(db)
        if frontier != plan["committed_rows"]:
            raise ValueError("classified committed frontier differs")
        with db:
            db.execute("UPDATE metadata SET value=? WHERE name='binding'", (json.dumps(binding, sort_keys=True),))
        campaign.immutable(directory / "binding.json", binding)
        with campaign.verify(original["records"]).open() as stream:
            for number, line in enumerate(stream, 1):
                digest = hashlib.sha256(line.encode()).hexdigest()
                if number <= frontier:
                    stored = db.execute("SELECT row_sha256 FROM completed WHERE number=?", (number,)).fetchone()
                    if stored is None or stored[0] != digest:
                        raise ValueError("consumed normalized row changed")
                    continue
                if committed >= plan["maximum_new_rows"]:
                    break
                if committed % 100 == 0:
                    work.checked(plan, "archive")
                budget.guard()
                row = json.loads(line)
                item = CachedRows(scope["campaign_root"], db, budget, physical, traces,
                                  row["kind"].startswith("focused-teacher"))
                item.trace(row["turns"], row["input"], row["locator"], row["kind"], row["group"])
                if any(item.counts[name] for name in census.FAILURES):
                    raise ValueError("normalized continuation failed rules")
                with db:
                    for category in ("states", "features", "groups"):
                        census.union(db, category, getattr(item, category))
                    if row["kind"].startswith("focused-teacher"):
                        census.union(db, "teacher:focused_observed_states", item.states)
                        census.union(db, "teacher:focused_observed_features", item.features)
                    db.execute("INSERT INTO completed VALUES (?,?,?,?)", (number, digest,
                        json.dumps(dict(item.counts)), json.dumps(item.ledger)))
                committed += 1
                reused += item.indexed_reuses
                if len(physical) > 100000:
                    physical.clear()
                if len(traces) > 25000:
                    traces.clear()
                if committed % 100 == 0:
                    campaign.atomic(output / "PROGRESS.json", dict(committed_rows=frontier + committed,
                        total_rows=original["rows"], new_rows=committed, indexed_state_reuses=reused))
        final = census.checkpoint(db)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        complete = final == original["rows"]
        if complete:
            receipt = census.export_inventory(db, directory / "inventory-001.json", binding, scope, closure,
                                              "complete", None, budget)
            receipt.update(schema=census.SCHEMA, checkpoint=str(database.resolve()),
                           historical_protected_reads=False, historical_ancestry_rescan=False,
                           games_run=0, new_committed_rows=committed, indexed_state_reuses=reused)
            campaign.immutable(directory / "receipt-001.json", receipt)
    finally:
        db.close(); source.close()
    # Recheck the parent's bound data/WAL after the continuation: no repair or
    # checkpointing of its historical bytes is permitted.
    campaign.verify(plan["parent_database"])
    if plan.get("parent_wal"):
        campaign.verify(plan["parent_wal"])
    result = dict(passed=True, complete=complete, plan=campaign.record(plan_path),
        database=campaign.record(database), binding=campaign.record(directory / "binding.json"),
        committed_rows=final, total_rows=original["rows"], new_rows=committed,
        indexed_state_reuses=reused, normalized_plan=plan["normalized_plan"],
        historical_protected_reads=False, historical_ancestry_rescan=False, games_run=0)
    if complete:
        frozen = dict(schema="papersoccer.rank-two.live-prior-exposure.v1", historical_protected_reads=False,
            packet_id="focused-network-exposure-continuation", extension_plan=campaign.record(plan_path),
            inputs=dict(database=result["database"], binding=result["binding"], inventory=receipt["inventory"],
                        receipt=campaign.record(directory / "receipt-001.json"), closure=original["closure"], scope=original["scope"]))
        with validator.prior_census(frozen):
            pass
        campaign.immutable(output / "bank-prior.json", frozen)
        result.update(prior=campaign.record(output / "bank-prior.json"), exposures=original["exposures"])
        campaign.immutable(output / "ready.json", result)
    campaign.immutable(output / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resume(args.plan)))


if __name__ == "__main__":
    main()
