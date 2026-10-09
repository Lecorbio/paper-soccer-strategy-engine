"""Initialize one inherited SQLite base without replaying new exposure rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_exposure_v4 as retained
from tools import rank_two_focused_work_v1 as work

SCHEMA = campaign.SCHEMA + ".census-zero-seed.v1"


def run(plan_path):
    plan=campaign.read(plan_path)
    work.checked(plan,"archive",launch=True)
    if plan["producer"]!=campaign.record(__file__):
        raise ValueError("zero-row seed producer differs")
    for ref in plan["inputs"].values():campaign.verify(ref)
    original=campaign.read(campaign.verify(plan["normalized_plan"]))
    if original["producer"]!=campaign.record(retained.__file__):
        raise ValueError("frozen normalization producer differs")
    prior=campaign.read(retained.verify(original["seed_prior"]))
    if prior["inputs"]!=original["seed_inputs"]:
        raise ValueError("inherited prior binding differs")
    with retained.prior_validator.prior_census(prior):pass
    out=Path(plan["output"]);directory=out/"census";directory.mkdir(parents=True,exist_ok=True)
    campaign.immutable(out/"CLAIM.json",dict(plan=campaign.record(plan_path),committed_rows=0,
        inherited_database=prior["inputs"]["database"],new_rows_replayed=0))
    binding=dict(schema=retained.census.SCHEMA,scope=original["scope"],closure=original["closure"],
        records=original["records"],sources=original["producers"],
        inherited_inputs=[prior["inputs"]["inventory"]],
        indexed_seed=dict(database=prior["inputs"]["database"],receipt=prior["inputs"]["receipt"],
                          plan=plan["normalized_plan"]),extension_producer=original["producer"])
    binding_ref=campaign.immutable(directory/"binding.json",binding)
    database=directory/"checkpoint.sqlite3"
    source=sqlite3.connect(Path(prior["inputs"]["database"]["path"]).resolve().as_uri()+"?mode=ro&immutable=1",uri=True)
    target=sqlite3.connect(database)
    try:
        for connection in (source,target):
            connection.execute("PRAGMA cache_size=-32768");connection.execute("PRAGMA mmap_size=0")
        source.backup(target,pages=256)
        with target:
            target.execute("DELETE FROM completed");target.execute("DELETE FROM metadata")
            bootstrap=dict(completed=0,historical_counts_unconfirmed={},
                inherited_inventories=[dict(kind="exposure",input=prior["inputs"]["inventory"])],
                indexed_seed_receipt=prior["inputs"]["receipt"])
            target.execute("INSERT INTO metadata VALUES (?,?)",("bootstrap",json.dumps(bootstrap,sort_keys=True)))
            target.execute("INSERT INTO metadata VALUES (?,?)",("binding",json.dumps(binding,sort_keys=True)))
        if retained.census.checkpoint(target)!=0:
            raise ValueError("zero-row committed frontier differs")
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        target.close();source.close()
    campaign.verify(prior["inputs"]["database"])
    result=dict(schema=SCHEMA,passed=True,plan=campaign.record(plan_path),
        normalized_plan=plan["normalized_plan"],database=campaign.record(database),binding=binding_ref,
        committed_rows=0,total_rows=original["rows"],new_rows_replayed=0,
        inherited_database_retained=prior["inputs"]["database"],
        historical_ancestry_rescan=False,historical_protected_reads=False,new_games=0,
        initialization_matches_retained_exposure_bootstrap=True)
    campaign.immutable(out/"RESULT.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--plan",type=Path,required=True)
    args=parser.parse_args();result=run(args.plan)
    print(json.dumps({k:result[k] for k in ("passed","committed_rows","total_rows")}))


if __name__=="__main__":main()
