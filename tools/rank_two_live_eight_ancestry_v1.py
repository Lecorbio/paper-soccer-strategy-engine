"""Validate and extend the retained full census plus immutable SQLite deltas."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_focused_prior_v1 as prior
from tools import rank_two_focused_census_v4 as delta
from tools import rank_two_focused_exposure_v4 as normalize
from tools import rank_two_focused_work_v1 as work


@contextmanager
def databases(state):
    union=c.read(c.verify(state['focused_terminal_ancestry_union']))
    if not union.get('passed') or union.get('fresh_bank_admission') is not False:
        raise ValueError('retained terminal union differs')
    ready=c.read(c.verify(union['parent_ready']));frozen=c.read(c.verify(ready['prior']))
    with prior.prior_census(frozen) as parent:
        layers=[]
        try:
            layers.append(delta.closed_database(union['overlay']))
            for ref in state.get('live_eight_census_layers',[]):
                receipt=c.read(c.verify(ref))
                if not receipt.get('passed') or receipt['original_union']!=state['focused_terminal_ancestry_union']:
                    raise ValueError('incremental ancestry layer binding differs')
                layers.append(delta.closed_database(receipt['database']))
            yield [*reversed(layers),parent]
        finally:
            for db in layers:db.close()


def validate(base,output):
    base=Path(base);state=c.read(base/'CURRENT.json')
    with databases(state) as dbs:
        # Check delta integrity only. The full prior's saved validator verifies
        # its metadata/frontier/hash; no ancestral trajectories are replayed.
        for db in dbs[:-1]:
            if db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('delta SQLite integrity differs')
    return c.immutable(output,dict(schema=c.SCHEMA+'.layered-ancestry',passed=True,
        original_union=state['focused_terminal_ancestry_union'],layers=state.get('live_eight_census_layers',[]),
        protected_gameplay_reads=False,historical_trajectory_rescan=False,read_only_parent=True))


def extend(plan_path):
    plan=c.read(plan_path);state=work.checked(plan,'archive',launch=True);out=Path(plan['output'])
    if plan['producer']!=c.record(__file__):raise ValueError('ancestry producer changed')
    normalized=normalize.prepare(plan['base'],state['focused_ancestry_ready']['path'],plan['exposures'],out/'normalized')
    scope=c.read(c.verify(normalized['scope']));database=out/'delta.sqlite3'
    if database.exists():raise RuntimeError('claimed ancestry delta exists; classify rather than replay')
    budget=delta.census.Budget(plan['resource_budget']['wall_seconds'],plan['resource_budget']['rss_bytes'])
    db=sqlite3.connect(database)
    try:
        db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA cache_size=-16384')
        db.executescript('''CREATE TABLE keys(category TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(category,value)) WITHOUT ROWID;
            CREATE TABLE completed(number INTEGER PRIMARY KEY,row_sha256 TEXT NOT NULL,counts TEXT NOT NULL,ledger TEXT NOT NULL);
            CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL);''')
        binding=dict(plan=c.record(plan_path),original_union=state['focused_terminal_ancestry_union'],parents=state.get('live_eight_census_layers',[]),normalized_plan=c.record(out/'normalized/PLAN.json'))
        with db:db.execute('INSERT INTO metadata VALUES (?,?)',('binding',json.dumps(binding,sort_keys=True)))
        count=0;physical={};anchor=None
        with databases(state) as parents:
            lookup=delta.Lookup([db,*parents])
            with c.verify(normalized['records']).open() as stream:
                for number,line in enumerate(stream,1):
                    work.checked(plan,'archive');row=json.loads(line)
                    if row['mode']!='trace':raise ValueError('explicit normalized trace required')
                    item=delta.DeltaRows(scope['campaign_root'],lookup,budget,physical,anchor,row['kind'].startswith('focused-teacher'))
                    item.trace(row['turns'],row['input'],row['locator'],row['kind'],row['group'])
                    if any(item.counts[name] for name in delta.census.FAILURES):
                        raise ValueError('trace requires explicit partial/failure classification')
                    delta.insert_row(db,item,number,hashlib.sha256(line.encode()).hexdigest())
                    anchor=item.publish_committed(physical);count+=1
        if count!=normalized['rows']:raise ValueError('new normalized frontier differs')
        counts=dict(db.execute('SELECT category,count(*) FROM keys GROUP BY category'))
    finally:db.close()
    result=c.immutable(out/'RESULT.json',dict(schema=c.SCHEMA+'.ancestry-delta',passed=True,
        database=c.record(database),original_union=state['focused_terminal_ancestry_union'],binding=binding,
        committed_rows=count,key_counts=counts,exposures=plan['exposures'],no_parent_database_copy=True,
        no_historical_trajectory_rescan=True,new_games=0,training_admission=False))
    with c.locked(plan['base']):
        current=c.read(Path(plan['base'])/'CURRENT.json');current.setdefault('live_eight_census_layers',[]).append(result)
        covered=set(r['sha256'] for r in plan['exposures'])
        current['pending_exposures']=[r for r in current.get('pending_exposures',[]) if r['sha256'] not in covered]
        c.atomic(Path(plan['base'])/'CURRENT.json',current)
    admitted=validate(plan['base'],out/'VALIDATED_UNION.json')
    with c.locked(plan['base']):
        current=c.read(Path(plan['base'])/'CURRENT.json');current['live_eight_ancestry_validated']=admitted;c.atomic(Path(plan['base'])/'CURRENT.json',current)
    return dict(passed=True,rows=count,receipt=result,validated_union=admitted)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);a=p.parse_args();print(json.dumps(extend(a.plan)))
