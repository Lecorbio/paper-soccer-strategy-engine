"""Retain immutable ancestry deltas and merge one compatible final index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3

from tools import rank_two_focused_campaign_v1 as campaign
from tools import rank_two_focused_census_v3 as prefix
from tools import rank_two_focused_prior_v1 as validator
from tools import rank_two_focused_work_v1 as work

census = prefix.census
SCHEMA = campaign.SCHEMA + '.ancestry-deltas.v1'


def closed_database(reference):
    path = campaign.verify(reference)
    wal = Path(str(path) + '-wal')
    if wal.exists() and wal.stat().st_size:
        raise ValueError('immutable input database has an unbound WAL')
    connection = sqlite3.connect(path.resolve().as_uri() + '?mode=ro&immutable=1', uri=True)
    connection.execute('PRAGMA cache_size=-32768')
    connection.execute('PRAGMA mmap_size=0')
    return connection


class Lookup:
    def __init__(self, databases):
        self.databases = databases

    def execute(self, sql, arguments=()):
        for db in self.databases:
            cursor = db.execute(sql, arguments)
            row = cursor.fetchone()
            if row is not None:
                return Answer(row)
        return Answer(None)


class Answer:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class DeltaRows(prefix.PrefixRows):
    def observe(self, state):
        self.observations += 1
        if self.observations % 256 == 0:
            self.guard()
        physical = (state.ball, state.to_move, state.winner,
                    sum(census.v2.EDGE_BITS[edge] for edge in state.used_segments))
        entry = self.pending_physical.get(physical, self.committed_physical.get(physical))
        canonical, feature = entry if entry is not None else (census.e.fingerprint(state), None)
        if feature is None:
            if not self.teacher and self.db.execute(
                    "SELECT 1 FROM keys WHERE category='states' AND value=?", (canonical,)).fetchone():
                self.indexed_reuses += 1
            else:
                feature = census.banks.feature_key(state)
        if feature is not None:
            self.states.add(canonical)
            self.features.add(feature)
            self.feature_by_state[canonical] = feature
        self.pending_physical[physical] = (canonical, feature)
        return canonical


def bound_inputs(plan):
    if (plan['producer'] != campaign.record(__file__)
            or plan['prefix_producer'] != campaign.record(prefix.__file__)):
        raise ValueError('delta/prefix producer changed')
    work.checked(plan, 'archive', launch=True)
    original = campaign.read(campaign.verify(plan['normalized_plan']))
    for name in ('records', 'scope', 'closure', 'producer', 'teacher_validator',
                 'proposal_algorithm', 'prior_validator'):
        campaign.verify(original[name])
    for ref in original['producers'].values():
        campaign.verify(ref)
    closure = campaign.read(original['closure']['path'])
    for ref in closure['inputs']:
        campaign.verify(ref)
    base = campaign.read(campaign.verify(plan['base_result']))
    if (base.get('passed') is not True or base['normalized_plan'] != plan['normalized_plan']
            or not 0 <= base['committed_rows'] <= original['rows']):
        raise ValueError('full base frontier or normalization differs')
    campaign.verify(base['database'])
    binding = campaign.read(campaign.verify(base['binding']))
    if binding['records'] != original['records']:
        raise ValueError('full base database is not bound to the normalized records')
    frontier, deltas = base['committed_rows'], []
    for reference in plan['delta_results']:
        delta = campaign.read(campaign.verify(reference))
        if (delta.get('passed') is not True or delta.get('schema') != SCHEMA + '.delta'
                or delta['base_result'] != plan['base_result']
                or delta['normalized_plan'] != plan['normalized_plan']
                or delta['start_row'] != frontier or delta['committed_rows'] < frontier):
            raise ValueError('delta chain has a gap, overlap or changed base')
        recipe = campaign.read(campaign.verify(delta['plan']))
        if recipe['producer'] != plan['producer']:
            raise ValueError('delta chain producer differs')
        campaign.verify(delta['database'])
        campaign.verify(delta['binding'])
        frontier = delta['committed_rows']
        deltas.append(delta)
    if frontier != plan['committed_rows']:
        raise ValueError('declared frontier differs from the sealed chain')
    return original, base, deltas, frontier


def new_claim(plan_path, plan, kind):
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'CLAIM.json').exists():
        raise RuntimeError('unknown spent delta claim retained; classify before continuing')
    campaign.immutable(output / 'CLAIM.json', dict(plan=campaign.record(plan_path), kind=kind,
        frontier=plan['committed_rows']))
    return output


def insert_row(db, item, number, digest):
    with db:
        for name in ('states', 'features', 'groups'):
            census.union(db, name, getattr(item, name))
        if item.teacher:
            census.union(db, 'teacher:focused_observed_states', item.states)
            census.union(db, 'teacher:focused_observed_features', item.features)
        db.execute('INSERT INTO completed VALUES (?,?,?,?)', (number, digest,
            json.dumps(dict(item.counts), sort_keys=True), json.dumps(item.ledger, sort_keys=True)))


def stage(plan_path):
    plan = campaign.read(plan_path)
    original, base, deltas, frontier = bound_inputs(plan)
    if not 1 <= plan['maximum_new_rows'] <= 500000 or frontier >= original['rows']:
        raise ValueError('bounded unspent normalized rows required')
    output = new_claim(plan_path, plan, 'delta')
    database = output / 'delta.sqlite3'
    db = sqlite3.connect(database)
    inherited = []
    budget = census.Budget(plan['cpu_seconds'], plan['maximum_rss_bytes'])
    binding = dict(schema=SCHEMA + '.binding', plan=campaign.record(plan_path),
        normalized_plan=plan['normalized_plan'], records=original['records'],
        base_result=plan['base_result'], start_row=frontier, delta_results=plan['delta_results'])
    physical, anchor, count, reuses, indexed, reason, partial = {}, None, 0, 0, 0, None, None
    try:
        inherited.append(closed_database(base['database']))
        if census.checkpoint(inherited[0]) != base['committed_rows']:
            raise ValueError('full base committed rows differ')
        observed_base = json.loads(inherited[0].execute("SELECT value FROM metadata WHERE name='binding'").fetchone()[0])
        if observed_base != campaign.read(base['binding']['path']):
            raise ValueError('full base database metadata differs')
        for delta in deltas:
            inherited.append(closed_database(delta['database']))
            expected = campaign.read(delta['binding']['path'])
            observed = json.loads(inherited[-1].execute("SELECT value FROM metadata WHERE name='binding'").fetchone()[0])
            if expected != observed:
                raise ValueError('delta database metadata differs')
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-32768')
        db.executescript('''CREATE TABLE keys(category TEXT NOT NULL, value TEXT NOT NULL,
            PRIMARY KEY(category,value)) WITHOUT ROWID;
            CREATE TABLE completed(number INTEGER PRIMARY KEY, row_sha256 TEXT NOT NULL,
                counts TEXT NOT NULL, ledger TEXT NOT NULL);
            CREATE TABLE metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL);''')
        with db:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', json.dumps(binding, sort_keys=True)))
        campaign.immutable(output / 'binding.json', binding)
        lookup = Lookup([db, *reversed(inherited)])
        with campaign.verify(original['records']).open() as stream:
            for number, line in enumerate(stream, 1):
                if number <= frontier:
                    continue
                if count >= plan['maximum_new_rows']:
                    reason = 'declared row cap'
                    break
                try:
                    budget.guard()
                except (TimeoutError, MemoryError) as exc:
                    reason = str(exc)
                    break
                if count % 100 == 0:
                    work.checked(plan, 'archive')
                digest, row = hashlib.sha256(line.encode()).hexdigest(), json.loads(line)
                if row['mode'] != 'trace':
                    raise ValueError('normalized trace required')
                item = DeltaRows(campaign.read(original['scope']['path'])['campaign_root'], lookup, budget, physical, anchor,
                                 row['kind'].startswith('focused-teacher'))
                try:
                    item.trace(row['turns'], row['input'], row['locator'], row['kind'], row['group'])
                except (TimeoutError, MemoryError) as exc:
                    # Preserve every returned observation conservatively, but
                    # never count an uncommitted row as completed.
                    with db:
                        for name in ('states', 'features', 'groups'):
                            census.union(db, name, getattr(item, name))
                        if item.teacher:
                            census.union(db, 'teacher:focused_observed_states', item.states)
                            census.union(db, 'teacher:focused_observed_features', item.features)
                    partial = dict(number=number, row_sha256=digest, confirmed=False)
                    reason = str(exc)
                    break
                if any(item.counts[name] for name in census.FAILURES):
                    raise ValueError('normalized row failed rules')
                insert_row(db, item, number, digest)
                anchor = item.publish_committed(physical)
                count += 1
                reuses += item.prefix_reuses
                indexed += item.indexed_reuses
                if count % 100 == 0:
                    campaign.atomic(output / 'PROGRESS.json', dict(committed_rows=number,
                        total_rows=original['rows'], new_rows=count, prefix_reuses=reuses))
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        actual = db.execute('SELECT count(*),min(number),max(number) FROM completed').fetchone()
        if actual[0] != count or count and (actual[1] != frontier + 1 or actual[2] != frontier + count):
            raise ValueError('new committed rows are not contiguous')
    finally:
        db.close()
        for connection in inherited:
            connection.close()
    for ref in [base['database'], *[delta['database'] for delta in deltas], original['records']]:
        campaign.verify(ref)
    result = dict(schema=SCHEMA + '.delta', passed=True, complete=frontier + count == original['rows'],
        plan=campaign.record(plan_path), database=campaign.record(database),
        binding=campaign.record(output / 'binding.json'), normalized_plan=plan['normalized_plan'],
        base_result=plan['base_result'], start_row=frontier, committed_rows=frontier + count,
        total_rows=original['rows'], new_rows=count, prefix_reuses=reuses, indexed_state_reuses=indexed,
        stop_reason=reason, unconfirmed_row=partial, historical_ancestry_rescan=False,
        historical_protected_reads=False, games_run=0)
    campaign.immutable(output / 'RESULT.json', result)
    return result


def merge(plan_path):
    plan = campaign.read(plan_path)
    original, base, deltas, frontier = bound_inputs(plan)
    if frontier != original['rows']:
        raise ValueError('cannot publish an incomplete delta chain as ready')
    output = new_claim(plan_path, plan, 'merge')
    directory = output / 'census'
    directory.mkdir(exist_ok=True)
    database = directory / 'checkpoint.sqlite3'
    # One final copy; every input index and immutable delta remains intact.
    shutil.copyfile(campaign.verify(base['database']), database)
    db = sqlite3.connect(database, uri=True)
    budget = census.Budget(plan['cpu_seconds'], plan['maximum_rss_bytes'])
    binding = campaign.read(base['binding']['path'])
    binding = {**binding, 'storage_producer': plan['producer'], 'merge_plan': campaign.record(plan_path),
               'delta_results': plan['delta_results'], 'full_base_result': plan['base_result']}
    try:
        db.execute('PRAGMA cache_size=-32768')
        db.execute('PRAGMA mmap_size=0')
        for index, delta in enumerate(deltas):
            work.checked(plan, 'archive')
            budget.guard()
            alias = 'delta_' + str(index)
            db.execute('ATTACH DATABASE ? AS ' + alias,
                       (Path(delta['database']['path']).as_uri() + '?mode=ro&immutable=1',))
            observed = json.loads(db.execute('SELECT value FROM ' + alias + ".metadata WHERE name='binding'").fetchone()[0])
            if observed != campaign.read(delta['binding']['path']):
                raise ValueError('merge delta binding differs')
            with db:
                db.execute('INSERT OR IGNORE INTO keys SELECT * FROM ' + alias + '.keys')
                db.execute('INSERT INTO completed SELECT * FROM ' + alias + '.completed')
            db.execute('DETACH DATABASE ' + alias)
        if census.checkpoint(db) != frontier:
            raise ValueError('final merged frontier has a gap')
        with db:
            db.execute("UPDATE metadata SET value=? WHERE name='binding'", (json.dumps(binding, sort_keys=True),))
        campaign.immutable(directory / 'binding.json', binding)
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        receipt = census.export_inventory(db, directory / 'inventory-001.json', binding,
            campaign.read(original['scope']['path']), campaign.read(original['closure']['path']),
            'complete', None, budget)
        if receipt['status'] != 'complete':
            raise ValueError('merged ancestry receipt failed coverage')
        receipt.update(schema=census.SCHEMA, checkpoint=str(database.resolve()),
            historical_ancestry_rescan=False, historical_protected_reads=False, games_run=0)
        campaign.immutable(directory / 'receipt-001.json', receipt)
    finally:
        db.close()
    for ref in [base['database'], *[delta['database'] for delta in deltas], original['records']]:
        campaign.verify(ref)
    frozen = dict(schema='papersoccer.rank-two.live-prior-exposure.v1', historical_protected_reads=False,
        packet_id='focused-network-delta-merge', extension_plan=campaign.record(plan_path),
        inputs=dict(database=campaign.record(database), binding=campaign.record(directory / 'binding.json'),
            inventory=receipt['inventory'], receipt=campaign.record(directory / 'receipt-001.json'),
            closure=original['closure'], scope=original['scope']))
    with validator.prior_census(frozen):
        pass
    campaign.immutable(output / 'bank-prior.json', frozen)
    ready = dict(passed=True, plan=campaign.record(plan_path), prior=campaign.record(output / 'bank-prior.json'),
        inventory=receipt['inventory'], receipt=campaign.record(directory / 'receipt-001.json'),
        rows=frontier, exposures=original['exposures'], historical_ancestry_rescan=False,
        historical_protected_reads=False, games_run=0)
    campaign.immutable(output / 'ready.json', ready)
    campaign.immutable(output / 'RESULT.json', ready)
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    result = merge(args.plan) if args.merge else stage(args.plan)
    print(json.dumps({key:result[key] for key in ('passed', 'complete', 'committed_rows') if key in result}), flush=True)


if __name__ == '__main__':
    main()
