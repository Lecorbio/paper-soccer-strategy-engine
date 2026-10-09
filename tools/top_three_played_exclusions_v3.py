#!/usr/bin/env python3
"""Transactional, hash-bound continuation of normalized exposure census records.

Old partial sets are conservative unions. Old counters are historical only: v2
may have counted a parent or entire branch in the first unconfirmed row. New
counters and ledger entries belong exclusively to atomically completed v3 rows.
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import hashlib
import inspect
import json
from pathlib import Path
import resource
import signal
import sqlite3
import sys
import time
try:
    from . import top_three_played_exclusions_v2 as v2
except ImportError:
    import top_three_played_exclusions_v2 as v2

e, banks = v2.e, v2.banks
SCHEMA = 'papersoccer.top-three.resumable-exclusions.v3'
FAILURES = ('pilot_final_state_missing_from_original_inventory', 'invalid_branch_tails', 'incomplete_actions')


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def verify(ref):
    if record(ref['path']) != ref:
        raise ValueError('hash-bound identity mismatch: ' + ref['path'])
    return Path(ref['path'])


def producers():
    return {name: record(path) for name, path in {
        'exporter': __file__, 'v2': v2.__file__, 'legacy_adapter': v2.legacy.__file__,
        'features': e.rules.__file__, 'experiments': e.__file__, 'banks': banks.__file__,
        'campaign': e.campaign.__file__,
    }.items()}


def rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


class Budget:
    def __init__(self, seconds, rss_bytes):
        self.started = time.process_time()
        self.seconds, self.rss_bytes = seconds, rss_bytes

    def guard(self):
        if rss() > self.rss_bytes:
            raise MemoryError('census RSS cap reached')
        if time.process_time() - self.started > self.seconds - min(40, self.seconds / 10):
            raise TimeoutError('census CPU reserve reached')


class StoredSet:
    def __init__(self, db, name):
        self.db, self.name = db, name

    def __contains__(self, key):
        return self.db.execute('SELECT 1 FROM keys WHERE category=? AND value=?', (self.name, key)).fetchone() is not None


def union(db, name, values):
    db.executemany('INSERT OR IGNORE INTO keys VALUES (?,?)', ((name, value) for value in values))


class RowCensus(v2.Census):
    def __init__(self, root, db, budget):
        super().__init__(root, {'rss_bytes': budget.rss_bytes, 'cpu_seconds': budget.seconds})
        self.budget = budget
        self.pilot_states, self.pilot_features = StoredSet(db, 'pilot_states'), StoredSet(db, 'pilot_features')
        self.observations = 0

    def guard(self):
        self.budget.guard()

    def observe(self, state):
        self.observations += 1
        if self.observations % 256 == 0:
            self.guard()
        return super().observe(state)


def checkpoint(db):
    start = int(json.loads(db.execute("SELECT value FROM metadata WHERE name='bootstrap'").fetchone()[0])['completed'])
    rows = db.execute('SELECT number FROM completed ORDER BY number')
    for expected, (number,) in enumerate(rows, start + 1):
        if number != expected:
            raise ValueError('noncontiguous completed-row checkpoints')
        start = number
    return start


def initialize(db, binding, scope, closure, bootstrap, budget):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS keys(category TEXT NOT NULL, value TEXT NOT NULL, PRIMARY KEY(category,value)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS completed(number INTEGER PRIMARY KEY, row_sha256 TEXT NOT NULL, counts TEXT NOT NULL, ledger TEXT NOT NULL);
    ''')
    existing = db.execute("SELECT value FROM metadata WHERE name='binding'").fetchone()
    if existing:
        if json.loads(existing[0]) != binding:
            raise ValueError('checkpoint binding changed (producer/input/bootstrap identity)')
        return
    # A killed bootstrap is all-or-nothing, just like a completed normalized row.
    with db:
        base = {'completed': 0, 'historical_counts_unconfirmed': {}, 'inherited_inventories': []}
        for spec in scope['inherit']:
            fn = banks.learning_inventory if spec['kind'] == 'learning' else banks.exposure_inventory
            values, identity = fn(spec['input']['path'], spec['input']['sha256'])
            for name in ('states', 'features'):
                union(db, name, values[name])
                if spec['kind'] == 'learning':
                    union(db, 'pilot_' + name, values[name])
            union(db, 'groups', values['clusters'])
            base['inherited_inventories'].append({'kind': spec['kind'], 'input': identity})
            del values
            budget.guard()
        if bootstrap:
            old = e.read(bootstrap)
            if (old.get('extension_schema') != 'papersoccer.top-three.cumulative-exclusions.v2'
                    or old['normalized_closure'] != binding['closure']
                    or old['scope_manifest'] != binding['scope']
                    or old['total_normalized_rows'] != closure['rows']
                    or old.get('historical_protected_reads') is not False
                    or old['state_function_sha256'] != hashlib.sha256(inspect.getsource(e.fingerprint).encode()).hexdigest()):
                raise ValueError('incompatible v2 bootstrap')
            for name, ref in old['sources'].items():
                if ref != binding['sources']['v2' if name == 'exporter' else name]:
                    raise ValueError('v2 bootstrap producer changed')
            base['completed'] = old['last_fully_processed_normalized_row']
            if not 0 <= base['completed'] <= closure['rows']:
                raise ValueError('invalid bootstrap frontier')
            base['historical_counts_unconfirmed'] = old['counts']
            # Explicitly never add these counts to v3 counters.
            for name, field in [('states', 'canonical_state_sha256'), ('features', 'canonical_feature_sha256'), ('groups', 'root_group_ids')]:
                union(db, name, old[field])
                del old[field]
                budget.guard()
            classes = e.read(verify(old['teacher_intermediate_classification']))
            for name, values in classes['sets'].items():
                union(db, 'teacher:' + name, values)
            del classes, old
        db.execute('INSERT INTO metadata VALUES (?,?)', ('bootstrap', json.dumps(base, sort_keys=True)))
        db.execute('INSERT INTO metadata VALUES (?,?)', ('binding', json.dumps(binding, sort_keys=True)))


def stream_array(stream, db, category):
    stream.write('[')
    first = True
    for (value,) in db.execute('SELECT value FROM keys WHERE category=? ORDER BY value', (category,)):
        if not first:
            stream.write(',')
        stream.write(json.dumps(value))
        first = False
    stream.write(']')


def export_inventory(db, output, binding, scope, closure, status, error, budget):
    base = json.loads(db.execute("SELECT value FROM metadata WHERE name='bootstrap'").fetchone()[0])
    counts = collections.Counter()
    for (raw,) in db.execute('SELECT counts FROM completed ORDER BY number'):
        counts.update(json.loads(raw))
    if status == 'complete' and any(counts[k] or base['historical_counts_unconfirmed'].get(k, 0) for k in FAILURES):
        status, error = 'validation-failed', 'counterfactual endpoint/rule validation failed'
    classes = output.with_suffix('.teacher-classes.json')
    with classes.open('x') as stream:
        stream.write('{"schema":"papersoccer.top-three.teacher-intermediate-coverage.v1","sets":{')
        names = [row[0] for row in db.execute("SELECT DISTINCT category FROM keys WHERE category LIKE 'teacher:%' ORDER BY category")]
        for index, name in enumerate(names):
            if index:
                stream.write(',')
            stream.write(json.dumps(name[len('teacher:'):]) + ':')
            stream_array(stream, db, name)
        stream.write('}}\n')
    ledger = output.with_suffix('.ledger.jsonl')
    with ledger.open('x') as stream:
        for number, raw in db.execute('SELECT number,ledger FROM completed ORDER BY number'):
            stream.write(json.dumps({'normalized_row': number, 'entries': json.loads(raw)}, sort_keys=True) + '\n')
    value = {'schema': v2.legacy.SCHEMA, 'extension_schema': SCHEMA,
        'campaign_root': scope['campaign_root'], 'status': status, 'error': error,
        'complete_coverage': status == 'complete', 'scope_manifest': binding['scope'],
        'normalized_closure': binding['closure'], 'cutoff': scope['cutoff'],
        'inputs': closure['inputs'] + [s['input'] for s in scope['inherit']],
        'sources': binding['sources'], 'binding': binding,
        'includes_primitive_states': True, 'includes_turn_boundaries': True,
        'includes_terminal_and_valid_operational_prefixes': True,
        'state_function_sha256': hashlib.sha256(inspect.getsource(e.fingerprint).encode()).hexdigest(),
        'counts': dict(counts), 'counts_scope': 'Only fully committed v3 rows after bootstrap frontier; never sum with historical counters.',
        'historical_counts_unconfirmed': base['historical_counts_unconfirmed'],
        'bootstrap_completed_rows': base['completed'],
        'last_fully_processed_normalized_row': checkpoint(db), 'total_normalized_rows': closure['rows'],
        'trajectory_ledger': record(ledger), 'teacher_intermediate_classification': record(classes),
        'historical_ledger': binding.get('bootstrap_ledger'), 'training_eligible': False,
        'historical_protected_reads': False, 'games_run': 0, 'confirmation_banks_generated': 0,
        'cpu_seconds': time.process_time() - budget.started, 'peak_rss_bytes': rss()}
    with output.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True)[:-1])
        for name, field in [('states', 'canonical_state_sha256'), ('features', 'canonical_feature_sha256'), ('groups', 'root_group_ids')]:
            stream.write(',' + json.dumps(field) + ':')
            stream_array(stream, db, name)
        stream.write('}\n')
    return {'inventory': record(output), 'status': status, 'error': error,
            'completed_rows': value['last_fully_processed_normalized_row'], 'total_rows': closure['rows'],
            'bootstrap_completed_rows': base['completed'],
            'states': db.execute("SELECT count(*) FROM keys WHERE category='states'").fetchone()[0],
            'features': db.execute("SELECT count(*) FROM keys WHERE category='features'").fetchone()[0],
            'counts': dict(counts), 'cpu_seconds': time.process_time() - budget.started, 'peak_rss_bytes': rss()}


def run(scope_path, expected, closure_path, directory, bootstrap=None, cpu_seconds=1200,
        rss_bytes=4*1024**3, stop_after=None, interrupt_hook=None):
    if not 0 < cpu_seconds <= 1200 or not 0 < rss_bytes <= 4*1024**3:
        raise ValueError('budget exceeds one-worker continuation cap')
    budget = Budget(cpu_seconds, rss_bytes)
    scope_ref = {'path': str(Path(scope_path).resolve()), 'sha256': expected}
    scope = e.read(verify(scope_ref))
    closure_ref = record(closure_path)
    closure = e.read(closure_path)
    if scope.get('schema') != v2.SCOPE_SCHEMA or scope.get('historical_protected_reads') is not False:
        raise ValueError('explicit unprotected v2 scope required')
    if closure['scope'] != scope_ref or closure.get('all_payload_inputs_hash_verified_before_census') is not True:
        raise ValueError('normalized scope/verification changed')
    verify(closure['producer'])
    verify(scope['cutoff'])
    records = verify(closure['records'])
    binding = {'schema': SCHEMA, 'scope': scope_ref, 'closure': closure_ref,
               'records': closure['records'], 'sources': producers(),
               'inherited_inputs': [spec['input'] for spec in scope['inherit']]}
    for ref in binding['inherited_inputs']:
        verify(ref)
    if bootstrap:
        binding['bootstrap'] = record(bootstrap)
        old = e.read(bootstrap)
        for key, field in [('bootstrap_ledger', 'trajectory_ledger'), ('bootstrap_classes', 'teacher_intermediate_classification')]:
            binding[key] = old[field]
            verify(old[field])
        del old
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Exclusive process ownership; flock is automatically released on a crash.
    import fcntl
    with (directory / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = directory / 'binding.json'
        if manifest.exists():
            if e.read(manifest) != binding:
                raise ValueError('immutable input/producer binding changed')
        else:
            e.emit(manifest, binding)
        frozen = directory / 'producers'
        frozen.mkdir(exist_ok=True)
        for name, ref in binding['sources'].items():
            destination = frozen / (name + '.py')
            if destination.exists():
                if record(destination)['sha256'] != ref['sha256']:
                    raise ValueError('immutable producer snapshot changed')
            else:
                verify(ref)
                e.campaign.immutable(destination, Path(ref['path']).read_bytes())
        with contextlib.closing(sqlite3.connect(directory / 'checkpoint.sqlite3')) as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=FULL')
            db.execute('PRAGMA cache_size=-32768')
            if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('corrupt checkpoint database')
            initialize(db, binding, scope, closure, bootstrap, budget)
            done = checkpoint(db)
            committed = 0
            status, error = 'complete', None
            try:
                with records.open('rb') as stream:
                    total = 0
                    for number, raw in enumerate(stream, 1):
                        total = number
                        digest = hashlib.sha256(raw).hexdigest()
                        if number <= done:
                            stored = db.execute('SELECT row_sha256 FROM completed WHERE number=?', (number,)).fetchone()
                            if stored and stored[0] != digest:
                                raise ValueError('completed row digest changed')
                            continue
                        budget.guard()
                        census = RowCensus(scope['campaign_root'], db, budget)
                        row = json.loads(raw)
                        if row['mode'] == 'trace':
                            census.trace(row['turns'], row['input'], row['locator'], row['kind'], row.get('group'))
                        elif row['mode'] == 'branches':
                            census.branches(row)
                        else:
                            raise ValueError('unknown normalized record mode')
                        budget.guard()
                        with db:
                            for name in ('states', 'features', 'groups'):
                                union(db, name, getattr(census, name))
                            for name, values in census.teacher_classes.items():
                                union(db, 'teacher:' + name, values)
                            if interrupt_hook:
                                interrupt_hook(number, census)
                            db.execute('INSERT INTO completed VALUES (?,?,?,?)',
                                       (number, digest, json.dumps(dict(census.counts), sort_keys=True), json.dumps(census.ledger, sort_keys=True)))
                        committed += 1
                        if committed % 25 == 0:
                            print(json.dumps({'stage': 'census-v3', 'completed_rows': number, 'total_rows': closure['rows'], 'cpu_seconds': time.process_time()-budget.started, 'rss_bytes': rss()}), flush=True)
                        del census
                        if stop_after is not None and committed >= stop_after:
                            raise InterruptedError('requested test checkpoint stop')
                    if total != closure['rows']:
                        raise ValueError('normalized row count mismatch')
            except (MemoryError, TimeoutError, InterruptedError, KeyboardInterrupt) as exc:
                status, error = 'resource-stopped', str(exc) or 'interrupted'
            # Reverify all immutable identities before labeling an export complete.
            for ref in [binding['scope'], binding['closure'], binding['records'], *binding['sources'].values(), *binding['inherited_inputs']]:
                verify(ref)
            # A killed export can leave files without its final receipt. Never
            # reuse an artifact name or mistake those partial files for success.
            attempt = 1
            while any(directory.glob(f'*-{attempt:03d}*')):
                attempt += 1
            output = directory / f'inventory-{attempt:03d}.json'
            result = export_inventory(db, output, binding, scope, closure, status, error, budget)
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            result.update({'schema': SCHEMA, 'workers': 1, 'rss_cap_bytes': rss_bytes, 'cpu_cap_seconds': cpu_seconds,
                           'new_committed_rows': committed, 'checkpoint': str((directory/'checkpoint.sqlite3').resolve()),
                           'historical_protected_reads': False, 'games_run': 0, 'confirmation_banks_generated': 0})
            e.emit(directory / f'receipt-{attempt:03d}.json', result)
            return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', type=Path, required=True)
    p.add_argument('--scope-sha256', required=True)
    p.add_argument('--normalized-closure', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--bootstrap', type=Path)
    p.add_argument('--cpu-seconds', type=float, default=1200)
    args = p.parse_args()
    # Hard process cap also covers bootstrap, hashing, and streaming exports.
    cap = max(1, int(args.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 1))
    if hasattr(resource, 'RLIMIT_AS') and sys.platform != 'darwin':
        resource.setrlimit(resource.RLIMIT_AS, (4*1024**3, 4*1024**3))
    def stop(signum, frame):
        raise KeyboardInterrupt('termination signal')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGXCPU, stop)
    print(json.dumps(run(args.scope, args.scope_sha256, args.normalized_closure, args.directory,
                         bootstrap=args.bootstrap, cpu_seconds=args.cpu_seconds), indent=2))


if __name__ == '__main__':
    main()
