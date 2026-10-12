"""Bind each continuation's completed TRAIN teacher batch to its exact trace cap."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
        if type(maximum_rows) is not int or not 0 <= maximum_rows <= 1000000:
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


def continuation_binding(plan, state):
    root = Path(plan['continuation_root']).resolve()
    match = re.fullmatch(r'live-data-continuation-([0-9]{2})', root.name)
    if not match or match[1] == '00':
        raise PermissionError('explicit continuation identity required')
    prefix = 'live_eight_continuation_' + match[1] + '_'
    for field in ('declaration', 'teacher_plan', 'teacher_complete'):
        plan_field = 'teacher_completion' if field == 'teacher_complete' else field
        if not plan.get(plan_field) or plan[plan_field] != state.get(prefix + field):
            raise PermissionError('current continuation declaration or teacher binding differs')
    return prefix


def validate_scope(plan, state):
    prefix = continuation_binding(plan, state)
    base = Path(plan['base']).resolve()
    root = Path(plan['continuation_root']).resolve()
    if (not base.is_relative_to(c.ROOT) or not root.is_relative_to(base / 'live-eight-v1')
            or not root.name.startswith('live-data-continuation-')
            or plan['scope_kind'] != 'completed-current-TRAIN-teacher'):
        raise PermissionError('bound current TRAIN teacher archive scope required')
    teacher_root = root / 'teacher256-v1'
    if ([Path(path).resolve() for path in plan['payload_roots']] != [teacher_root / 'execution']
            or [Path(ref['path']).resolve() for ref in plan['payload_files']]
                != [teacher_root / 'TEACHER_PLAN.json', teacher_root / 'RESERVED_EXPOSURE.json']):
        raise PermissionError('only exact current teacher payloads and plan may be parsed')
    expected = [teacher_root / name for name in ('RESERVED_EXPOSURE.json',
        'execution/RESERVED_EXPOSURE.json', 'execution/EXPOSURE_PENDING.json')]
    if [Path(ref['path']).resolve() for ref in plan['exposures']] != expected:
        raise PermissionError('all and only current teacher exposures required')
    pending = {ref['sha256'] for ref in state['pending_exposures']}
    if (any(ref['sha256'] not in pending for ref in plan['exposures'])
            or plan['prior_validated_union'] != state['live_eight_ancestry_validated']
            or plan['teacher_completion'] != state[prefix + 'teacher_complete']
            or plan['teacher_plan'] != state[prefix + 'teacher_plan']):
        raise PermissionError('completed pending batch and current full prior binding required')
    for ref in plan['exposures'] + plan['payload_files']:
        c.verify(ref)
    completion = c.read(c.verify(plan['teacher_completion']))
    teacher_plan = c.read(c.verify(plan['teacher_plan']))
    c.verify(plan['declaration'])
    if teacher_plan.get('inputs', {}).get('declaration') != plan['declaration']:
        raise PermissionError('teacher batch belongs to a different frozen declaration')
    execution = c.read(c.verify(completion['result']))
    counts = c.read(c.verify(plan['payload_counts']))
    expected_rows = 2 * counts['parent_receipts'] + counts['successor_traces']
    if (not completion.get('passed') or not execution.get('passed')
            or completion['payload_counts'] != plan['payload_counts']
            or counts['execution'] != completion['result']
            or counts['teacher_plan'] != plan['teacher_plan']
            or execution['plan'] != plan['teacher_plan']
            or execution['requested_parents'] != execution['valid_labels']
            or execution['unsupported_labels'] != 0
            or len(teacher_plan['positions']) != counts['parent_receipts']
            or len(execution['parent_receipts']) != counts['parent_receipts']
            or counts['plan_parent_traces'] != counts['parent_receipts']
            or counts['expected_normalized_rows'] != expected_rows
            or plan['maximum_normalized_rows'] != expected_rows
            or not 0 < expected_rows <= 1000000
            or teacher_plan['nodes'] != 256000
            or teacher_plan['outcomes_as_targets'] is not False
            or any(row['split'] != 'train' for row in teacher_plan['positions'])):
        raise PermissionError('exact complete TRAIN teacher trace count and recipe required')
    return expected_rows


def extend(plan_path):
    plan = c.read(plan_path)
    state = work.checked(plan, 'archive', launch=True)
    out = Path(plan['output'])
    if plan['producer'] != c.record(__file__):
        raise ValueError('scoped ancestry producer changed')
    validate_scope(plan, state)
    normalized = prepare_scoped(plan, state, out / 'normalized')
    if normalized['rows'] != plan['maximum_normalized_rows']:
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
