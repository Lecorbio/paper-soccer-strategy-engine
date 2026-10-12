"""Close each declaration-bound frozen validation bank and its exact teacher exposures."""
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
        if type(maximum_rows) is not int or not 0 <= maximum_rows <= 100000:
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
    root = Path(plan['validation_root']).resolve()
    match = re.fullmatch(r'live-data-continuation-([0-9]{2})', root.parent.name)
    if not match or match[1] == '00' or root.name != 'fresh-validation-v1':
        raise PermissionError('explicit fixed validation continuation identity required')
    prefix = 'live_eight_continuation_' + match[1] + '_'
    fields = dict(declaration='declaration', generation_plan='validation_plan',
                  generation_completion='validation_complete', teacher_plan='validation_teacher_plan',
                  teacher_completion='validation_teacher_complete')
    for field, current_field in fields.items():
        if not plan.get(field) or plan[field] != state.get(prefix + current_field):
            raise PermissionError('current validation declaration or completed bank binding differs')
    declaration = c.read(c.verify(plan['declaration']))
    recipe = declaration.get('fresh_validation_recipe', {})
    seed = recipe.get('seed')
    expected = dict(colors=[0, 1], maximum_early_parents_per_root=2,
        minimum_comparable_early_roots=100, no_seed_or_root_replacement=True,
        outcomes_as_targets=False, root_depths=[8, 10], roots=128, seed=seed)
    if (type(seed) is not int or not 0 <= seed < 2**32 or recipe != expected
            or declaration.get('architecture') != [6301, 8, 8, 1]
            or declaration.get('outcomes_as_targets') is not False):
        raise PermissionError('frozen128-root/100early validation recipe required')
    return prefix, seed, match[1]


def declared_count(document, first, legacy):
    if first in document and legacy in document and document[first] != document[legacy]:
        raise PermissionError('validation completion count aliases disagree')
    value = document.get(first, document.get(legacy))
    if type(value) is not int:
        raise PermissionError('integer validation completion count required')
    return value


def validate_scope(plan, state):
    prefix, seed, identifier = continuation_binding(plan, state)
    base = Path(plan['base']).resolve()
    root = Path(plan['validation_root']).resolve()
    if (not base.is_relative_to(c.ROOT)
            or root != base / ('live-eight-v1/live-data-continuation-' + identifier) / 'fresh-validation-v1'
            or plan['scope_kind'] != 'completed-current-fixed-validation'):
        raise PermissionError('exact current fixed validation bank scope required')
    teacher = root / 'teacher256-v1'
    if ([Path(path).resolve() for path in plan['payload_roots']]
            != [root / 'games', teacher]
            or [Path(ref['path']).resolve() for ref in plan['payload_files']]
                != [root / 'PLAN.json', root / 'EXPOSURE_PENDING.json', root / 'COMPLETE_EXPOSURE.json']):
        raise PermissionError('only current validation games, labels and exact plan files may be parsed')
    expected = [root / 'EXPOSURE_PENDING.json', root / 'COMPLETE_EXPOSURE.json',
        teacher / 'RESERVED_EXPOSURE.json', teacher / 'execution/RESERVED_EXPOSURE.json',
        teacher / 'execution/EXPOSURE_PENDING.json']
    if [Path(ref['path']).resolve() for ref in plan['exposures']] != expected:
        raise PermissionError('all and only current fixed validation exposures required')
    pending = {ref['sha256'] for ref in state['pending_exposures']}
    if (any(ref['sha256'] not in pending for ref in plan['exposures'])
            or plan['prior_validated_union'] != state['live_eight_ancestry_validated']
            or plan['generation_completion'] != state[prefix + 'validation_complete']
            or plan['teacher_completion'] != state[prefix + 'validation_teacher_complete']):
        raise PermissionError('completed pending bank and current full prior required')
    for ref in plan['exposures'] + plan['payload_files']:
        c.verify(ref)
    generation = c.read(c.verify(plan['generation_completion']))
    completed = c.read(c.verify(plan['teacher_completion']))
    if (not generation.get('passed') or declared_count(generation, 'complete_games', 'games') != 256
            or declared_count(generation, 'frozen_roots', 'roots') != 128
            or generation['seed'] != seed or generation.get('plan') != plan['generation_plan']
            or completed.get('teacher_plan') != plan['teacher_plan'] or not completed.get('passed')
            or completed['validation_roots'] != 128 or not completed['validation_only']
            or completed['outcomes_as_targets'] is not False
            or not 0 < plan['maximum_normalized_rows'] <= 100000):
        raise PermissionError('frozen128-root/256-game validation-only evidence required')


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
