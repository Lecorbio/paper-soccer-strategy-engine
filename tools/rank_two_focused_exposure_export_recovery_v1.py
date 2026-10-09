"""Finish a committed ancestry export without copying or replaying its database."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from tools import rank_two_focused_exposure_v4 as retained
from tools import rank_two_focused_work_v1 as work

campaign, census = retained.campaign, retained.census
SCHEMA = campaign.SCHEMA + ".exposure-export-recovery.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    work.checked(plan, "archive", launch=True)
    if plan["producer"] != campaign.record(__file__):
        raise ValueError("export recovery producer changed")
    for reference in plan["inputs"].values():
        campaign.verify(reference)
    original = campaign.read(campaign.verify(plan["original_plan"]))
    binding = campaign.read(campaign.verify(plan["binding"]))
    scope = campaign.read(campaign.verify(original["scope"]))
    closure = campaign.read(campaign.verify(original["closure"]))
    if (binding["records"] != original["records"] or binding["closure"] != original["closure"]
            or binding["scope"] != original["scope"]):
        raise ValueError("committed database export binding differs")
    output = Path(plan["output"])
    campaign.immutable(output / "CLAIM.json", dict(plan=campaign.record(plan_path),
        database=plan["database"], rows_replayed=0, games_run=0))
    database = campaign.verify(plan["database"])
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA cache_size=-32768")
        db.execute("PRAGMA mmap_size=0")
        if campaign.read(campaign.verify(plan["original_plan"])) != original:
            raise ValueError("original plan changed")
        stored = json.loads(db.execute("SELECT value FROM metadata WHERE name='binding'").fetchone()[0])
        if stored != binding:
            raise ValueError("database binding changed")
        committed = list(db.execute("SELECT * FROM completed ORDER BY number"))
        if len(committed) != original["rows"] or census.checkpoint(db) != original["rows"]:
            raise ValueError("export-only recovery requires every row already committed")
        with campaign.verify(original["records"]).open() as stream:
            for number, line in enumerate(stream, 1):
                row = committed[number - 1]
                if row[0] != number or row[1] != hashlib.sha256(line.encode()).hexdigest():
                    raise ValueError("committed normalized row identity differs")
            if number != original["rows"]:
                raise ValueError("normalized frontier changed")
        work.checked(plan, "archive")
        budget = census.Budget(1200, 4 * 1024**3)
        result = census.export_inventory(db, output / "inventory-001.json", binding,
                                         scope, closure, "complete", None, budget)
        if result["status"] != "complete" or result["completed_rows"] != original["rows"]:
            raise ValueError("recovered inventory is incomplete")
    result.update(schema=census.SCHEMA, checkpoint=str(database), historical_protected_reads=False,
                  historical_ancestry_rescan=False, games_run=0, rows_replayed=0,
                  workers=1, rss_cap_bytes=4 * 1024**3, cpu_cap_seconds=1200)
    receipt = campaign.immutable(output / "receipt-001.json", result)
    prior = dict(schema="papersoccer.rank-two.live-prior-exposure.v1", historical_protected_reads=False,
        packet_id="focused-network-exposure-export-recovery", extension_plan=plan["original_plan"],
        inputs=dict(inventory=result["inventory"], database=plan["database"], receipt=receipt,
                    binding=plan["binding"], closure=original["closure"], scope=original["scope"]))
    with retained.prior_validator.prior_census(prior):
        pass
    prior_ref = campaign.immutable(output / "bank-prior.json", prior)
    ready = dict(passed=True, plan=campaign.record(plan_path), original_plan=plan["original_plan"],
        prior=prior_ref, inventory=result["inventory"], receipt=receipt, rows=original["rows"],
        exposures=original["exposures"], games_run=0, rows_replayed=0,
        historical_protected_reads=False, historical_ancestry_rescan=False,
        partial_exports_retained=plan["partial_exports"], all_previous_databases_retained=True)
    campaign.immutable(output / "ready.json", ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.plan)
    print(json.dumps(dict(passed=result["passed"], rows=result["rows"], rows_replayed=0, games_run=0)))


if __name__ == "__main__":
    main()
