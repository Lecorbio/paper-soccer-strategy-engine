"""Index terminal-only exposures in a small compatible SQLite ancestry delta."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from tools import rank_two_focused_exposure_v4 as normalization
from tools import rank_two_focused_census_v4 as delta
from tools import rank_two_focused_work_v1 as work

campaign, census = normalization.campaign, normalization.census
SCHEMA = campaign.SCHEMA + ".terminal-overlay.v1"


def run(plan_path):
    plan = campaign.read(plan_path)
    current = work.checked(plan, "archive", launch=True)
    if not current.get("research_closed") or plan["producer"] != campaign.record(__file__):
        raise ValueError("terminal-only overlay admission required")
    for ref in plan["inputs"].values():
        campaign.verify(ref)
    seed = campaign.read(campaign.verify(plan["seed_ready"]))
    prior = campaign.read(campaign.verify(seed["prior"]))
    out = Path(plan["output"])
    campaign.immutable(out/"CLAIM.json", dict(plan=campaign.record(plan_path),
        prior=seed["prior"], database_copy=False, new_games=0))
    normalized = normalization.prepare(plan["base"], plan["seed_ready"]["path"],
        plan["exposures"], out/"normalized")
    if not 0 <= normalized["rows"] <= 16:
        raise ValueError("bounded passive terminal trace count exceeded")
    parent = delta.closed_database(prior["inputs"]["database"])
    database = out/"delta.sqlite3"
    db = sqlite3.connect(database)
    binding = dict(schema=SCHEMA+".binding", plan=campaign.record(plan_path),
        prior=seed["prior"], parent_database=prior["inputs"]["database"],
        normalized_plan=campaign.record(out/"normalized/PLAN.json"))
    budget = census.Budget(plan["resource_budget"]["wall_seconds"],
                          plan["resource_budget"]["rss_bytes"])
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA cache_size=-16384")
        db.executescript('''CREATE TABLE keys(category TEXT NOT NULL,value TEXT NOT NULL,
            PRIMARY KEY(category,value)) WITHOUT ROWID;
            CREATE TABLE completed(number INTEGER PRIMARY KEY,row_sha256 TEXT NOT NULL,
                counts TEXT NOT NULL,ledger TEXT NOT NULL);
            CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL);''')
        with db:
            db.execute("INSERT INTO metadata VALUES (?,?)",("binding",json.dumps(binding,sort_keys=True)))
        lookup = delta.Lookup([db,parent])
        physical, anchor, count = {}, None, 0
        scope = campaign.read(campaign.verify(normalized["scope"]))
        with campaign.verify(normalized["records"]).open() as stream:
            for number,line in enumerate(stream,1):
                work.checked(plan,"archive")
                row=json.loads(line)
                if row["mode"]!="trace" or row["kind"].startswith("focused-teacher"):
                    raise ValueError("terminal passive exclusion trace required")
                item=delta.DeltaRows(scope["campaign_root"],lookup,budget,physical,anchor,False)
                item.trace(row["turns"],row["input"],row["locator"],row["kind"],row["group"])
                if any(item.counts[name] for name in census.FAILURES):
                    raise ValueError("passive terminal trace failed normalization or rules")
                delta.insert_row(db,item,number,hashlib.sha256(line.encode()).hexdigest())
                anchor=item.publish_committed(physical)
                count+=1
        if count!=normalized["rows"]:
            raise ValueError("terminal overlay frontier differs")
        counts=dict(db.execute("SELECT category,count(*) FROM keys GROUP BY category"))
    finally:
        db.close();parent.close()
    campaign.verify(prior["inputs"]["database"])
    receipt=campaign.immutable(out/"RESULT.json",dict(schema=SCHEMA,passed=True,
        plan=campaign.record(plan_path),normalized_plan=campaign.record(out/"normalized/PLAN.json"),
        database=campaign.record(database),binding=binding,committed_rows=count,key_counts=counts,
        existing_census_preserved=plan["seed_ready"],exposures=plan["exposures"],
        union_query_method="rank_two_focused_census_v4.Lookup: delta plus read-only parent",
        full_legacy_export_not_replaced=True,fresh_bank_admission=False,
        research_remains_closed=True,no_parent_rows_or_keys_rescanned=True,
        new_games=0,new_teacher_queries=0,training_eligible=False,
        historical_protected_reads=False,all_previous_databases_retained=True))
    return dict(passed=True,receipt=receipt,rows=count,key_counts=counts)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    print(json.dumps(run(parser.parse_args().plan)))


if __name__=="__main__":
    main()
