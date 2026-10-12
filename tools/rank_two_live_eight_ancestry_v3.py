"""Extend only explicit new played windows; retain opaque metadata separately."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

from tools import rank_two_live_eight_ancestry_v1 as old

c, normalize, delta, work = old.c, old.normalize, old.delta, old.work
databases, validate = old.databases, old.validate
OriginalNormalizer = normalize.Normalizer


class ScopedNormalizer(OriginalNormalizer):
    def __init__(self, stream, roots, files, maximum_rows):
        super().__init__(stream)
        self.allowed_roots = tuple(Path(path).resolve() for path in roots)
        self.allowed_files = {Path(path).resolve() for path in files}
        self.maximum_rows = maximum_rows
        if type(maximum_rows) is not int or not 0 <= maximum_rows <= 10000:
            raise ValueError('explicit normalization row cap required')

    def bind(self, reference, parse=False):
        path = Path(reference['path']).resolve()
        if (parse and path.suffix in ('.json', '.jsonl') and path not in self.allowed_files
                and not any(path.is_relative_to(root) for root in self.allowed_roots)):
            raise PermissionError('payload is outside the explicit new exposure scope')
        return super().bind(reference, parse)

    def emit(self, *args, **kwargs):
        if self.rows >= self.maximum_rows:
            raise PermissionError('new normalized trace row cap reached before census')
        return super().emit(*args, **kwargs)


def prepare_scoped(plan, state, output):
    original = normalize.Normalizer
    normalize.Normalizer = lambda stream: ScopedNormalizer(stream, plan['payload_roots'],
        [reference['path'] for reference in plan['payload_files']], plan['maximum_normalized_rows'])
    try:
        return normalize.prepare(plan['base'], state['focused_ancestry_ready']['path'],
                                 plan['exposures'], output)
    finally:
        normalize.Normalizer = original


def validate_scope(plan, state):
    base = Path(plan['base']).resolve()
    if not base.is_relative_to(c.ROOT) or plan['scope_kind'] != 'declared-played-windows':
        raise PermissionError('bound played-window archive scope required')
    identifiers = plan['attempt_ids']
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError('explicit unique current window identifiers required')
    expected_roots = [base / 'live-eight-v1/attempts' / aid for aid in identifiers]
    if [Path(path).resolve() for path in plan['payload_roots']] != expected_roots or plan['payload_files']:
        raise PermissionError('only the exact declared window directories may be parsed')
    expected_exposures = [str(root / 'EXPOSURE.json') for root in expected_roots]
    if [str(Path(ref['path']).resolve()) for ref in plan['exposures']] != expected_exposures:
        raise PermissionError('all and only the declared played-window exposures required')
    pending = {ref['sha256'] for ref in state['pending_exposures']}
    for aid, ref in zip(identifiers, plan['exposures'], strict=True):
        if (aid not in state['live_eight_declared_windows']
                or not state['live_eight_known_windows'].get(aid, {}).get('all90_archived')
                or ref['sha256'] not in pending):
            raise PermissionError('current archived pending window required')
    if plan['prior_validated_union'] != state['live_eight_ancestry_validated']:
        raise PermissionError('current full prior ancestry binding required')


def extend(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'archive', launch=True)
    out = Path(plan['output'])
    if plan['producer'] != c.record(__file__):
        raise ValueError('scoped ancestry producer changed')
    validate_scope(plan, state)
    normalized = prepare_scoped(plan, state, out / 'normalized')
    if normalized['rows'] > plan['maximum_normalized_rows']:
        raise PermissionError('normalized row cap differs')
    scope = c.read(c.verify(normalized['scope']))
    database = out / 'delta.sqlite3'
    if database.exists():
        raise RuntimeError('spent scoped census must be classified rather than replayed')
    budget = delta.census.Budget(plan['resource_budget']['wall_seconds'], plan['resource_budget']['rss_bytes'])
    db = sqlite3.connect(database)
    try:
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-16384')
        db.executescript('''CREATE TABLE keys(category TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(category,value)) WITHOUT ROWID;
            CREATE TABLE completed(number INTEGER PRIMARY KEY,row_sha256 TEXT NOT NULL,counts TEXT NOT NULL,ledger TEXT NOT NULL);
            CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT NOT NULL);''')
        binding = dict(plan=c.record(plan_path), original_union=state['focused_terminal_ancestry_union'],
            parents=state['live_eight_census_layers'], normalized_plan=c.record(out / 'normalized/PLAN.json'))
        with db:
            db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', json.dumps(binding, sort_keys=True)))
        count, physical, anchor = 0, {}, None
        with databases(state) as parents:
            lookup = delta.Lookup([db, *parents])
            with c.verify(normalized['records']).open() as stream:
                for number, line in enumerate(stream, 1):
                    work.checked(plan, 'archive')
                    row = json.loads(line)
                    if row['mode'] != 'trace':
                        raise ValueError('explicit normalized trace required')
                    item = delta.DeltaRows(scope['campaign_root'], lookup, budget, physical, anchor,
                                           row['kind'].startswith('focused-teacher'))
                    item.trace(row['turns'], row['input'], row['locator'], row['kind'], row['group'])
                    if any(item.counts[name] for name in delta.census.FAILURES):
                        raise ValueError('trace requires partial/failure classification')
                    delta.insert_row(db, item, number, hashlib.sha256(line.encode()).hexdigest())
                    anchor = item.publish_committed(physical)
                    count += 1
        if count != normalized['rows']:
            raise ValueError('new normalized frontier differs')
        counts = dict(db.execute('SELECT category,count(*) FROM keys GROUP BY category'))
    finally:
        db.close()
    result = c.immutable(out / 'RESULT.json', dict(schema=c.SCHEMA + '.ancestry-delta', passed=True,
        database=c.record(database), original_union=state['focused_terminal_ancestry_union'],
        binding=binding, committed_rows=count, key_counts=counts, exposures=plan['exposures'],
        no_parent_database_copy=True, no_historical_trajectory_rescan=True,
        normalized_row_cap=plan['maximum_normalized_rows'], explicit_payload_scope=True,
        new_games=0, training_admission=False))
    with c.locked(plan['base']):
        current = c.read(Path(plan['base']) / 'CURRENT.json')
        current['live_eight_census_layers'].append(result)
        covered = {reference['sha256'] for reference in plan['exposures']}
        current['pending_exposures'] = [reference for reference in current['pending_exposures']
                                        if reference['sha256'] not in covered]
        c.atomic(Path(plan['base']) / 'CURRENT.json', current)
    union = validate(plan['base'], out / 'VALIDATED_UNION.json')
    with c.locked(plan['base']):
        current = c.read(Path(plan['base']) / 'CURRENT.json')
        current['live_eight_ancestry_validated'] = union
        c.atomic(Path(plan['base']) / 'CURRENT.json', current)
    return dict(passed=True, rows=count, receipt=result, validated_union=union)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(extend(parser.parse_args().plan)))
