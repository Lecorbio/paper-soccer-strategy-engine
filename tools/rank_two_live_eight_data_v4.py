"""Bound new live TRAIN labels without changing retained data producers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from tools import rank_two_live_eight_data_v2 as prior

c, data, work = prior.c, prior.data, prior.work
SCHEMA = prior.SCHEMA
ROOT_CAP, PARENTS_PER_ROOT = 64, 16


def bounded_parents(rows, seed, root_cap=ROOT_CAP, parents_per_root=PARENTS_PER_ROOT):
    """Sample canonical identities; outcomes never enter the ordering."""
    if type(seed) is not int or not 1 <= root_cap <= ROOT_CAP or not 1 <= parents_per_root <= PARENTS_PER_ROOT:
        raise ValueError('invalid frozen label sampling bounds')
    grouped = defaultdict(dict)
    for row in rows:
        if row['split'] != 'train' or row.get('outcome_as_target') is not False:
            raise PermissionError('only alternative-label TRAIN parents may be sampled')
        key = (row['canonical_state'], row['canonical_features'])
        previous = grouped[row['root_group_id']].get(key)
        if previous is None or row['position_id'] < previous['position_id']:
            grouped[row['root_group_id']][key] = row
    def rank(value):
        return hashlib.sha256((str(seed) + ':' + value).encode('ascii')).hexdigest(), value
    roots = sorted(grouped, key=rank)[:root_cap]
    selected = []
    for root in roots:
        positions = sorted(grouped[root].values(), key=lambda row: rank(
            row['canonical_state'] + ':' + row['canonical_features']))
        selected.extend(positions[:parents_per_root])
    return selected


def merge_live_roots(old_live, new_live, train_roots):
    live = set(old_live) | set(new_live)
    if not new_live or not live <= set(train_roots):
        raise ValueError('live TRAIN root provenance is incomplete')
    return sorted(live)


def prepare(plan_path):
    plan = c.read(plan_path)
    state = prior.check_current(plan, 'label', launch=True)
    c.resource_guard(plan['resource_budget'])
    if (plan['producer'] != c.record(__file__) or plan['teacher_nodes'] != 256000
            or plan['sampling'] != dict(root_cap=ROOT_CAP, parents_per_root=PARENTS_PER_ROOT,
                                      ordering='seeded canonical identity; independent of outcomes')):
        raise ValueError('frozen bounded live-data recipe differs')
    if state.get('pending_exposures'):
        raise PermissionError('close prior exposures before preparing new labels')
    audit = c.read(c.verify(plan['prebank_live_check']))
    if not audit['passed'] or audit['unknown_game_ids']:
        raise PermissionError('current live metadata has unclosed exposures')
    union = c.read(c.verify(plan['ancestry']))
    if not union['passed'] or union['layers'] != state['live_eight_census_layers']:
        raise PermissionError('full prior and all incremental ancestry layers required')
    rows, game_attributes = [], {}
    for reference in plan['windows']:
        window = c.read(c.verify(reference))
        rows.extend(prior.parents(window, reference, plan['seed']))
        for game in window['games']:
            if c.training_allowed(window, game):
                game_attributes[str(game['game_id'])] = (game['color'], bool(game['won']))
    selected = bounded_parents(rows, plan['seed'])
    if not selected:
        raise ValueError('no eligible new live TRAIN parents')
    selected_games = {row['game_id'] for row in selected}
    coverage = Counter(game_attributes[gid] for gid in selected_games)
    if {color for color, _ in coverage} != {0, 1} or {won for _, won in coverage} != {False, True}:
        raise ValueError('frozen sample lacks both colors or both outcomes; never replace the seed silently')
    out = Path(plan['output'])
    positions = c.immutable(out / 'POSITIONS.jsonl', b''.join(
        (json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii') for row in selected))
    proof = c.immutable(out / 'PROVENANCE.json', dict(
        passed=True, roots=sorted({row['root_group_id'] for row in selected}), windows=plan['windows'],
        positions=positions, only_predesignated_new_exploration=True, protected_games_read=False,
        both_outcomes_eligible=True, outcomes_as_targets=False, teacher_nodes=256000,
        canonical_reflections_not_independent=True, grouping='canonical opening after four complete turns',
        selection_uses_game_strength=False, sampling=plan['sampling'], sampling_seed=plan['seed'],
        candidate_parent_count=len(rows), selected_parent_count=len(selected), selected_game_count=len(selected_games),
        color_outcome_game_counts=[dict(color=color, won=won, games=count)
                                  for (color, won), count in sorted(coverage.items())],
        native_teacher_seed_policy='accepted position-and-work binding; no pure-budget causal claim'))
    value = dict(passed=True, positions=positions, provenance=proof, parent_count=len(selected))
    c.immutable(out / 'RESULT.json', value)
    return value


def combined_live_provenance(old_roots, new_roots, old_proof, new_proof):
    for proof in (old_proof, new_proof):
        if (not proof.get('passed') or not proof.get('only_predesignated_new_exploration')
                or proof.get('protected_games_read') is not False):
            raise PermissionError('retained and new live provenance must both be eligible')
    if not set(old_roots) <= set(old_proof['roots']) or not set(new_roots) <= set(new_proof['roots']):
        raise ValueError('live root namespace escapes its eligible provenance')
    return dict(schema=SCHEMA + '.combined-live-provenance', passed=True,
                roots=sorted(set(old_roots) | set(new_roots)),
                retained_live_roots=sorted(old_roots), new_live_roots=sorted(new_roots),
                only_predesignated_new_exploration=True, protected_games_read=False,
                outcomes_as_targets=False, canonical_reflections_not_independent=True,
                grouping='canonical opening after four complete turns', teacher_nodes=256000)


def import_labels(teacher_plan_path, execution_path, retained_corpus, fresh_validation, output):
    """Retain all old TRAIN groups and the complete live root namespace."""
    plan = c.read(teacher_plan_path)
    prior.check_current(plan, 'label', launch=True)
    prior.verify_teacher_plan(plan)
    execution = c.read(execution_path)
    if not execution.get('passed') or execution['plan'] != c.record(teacher_plan_path):
        raise ValueError('teacher execution incomplete')
    old, fresh = c.read(c.verify(retained_corpus)), c.read(c.verify(fresh_validation))
    if (fresh.get('passed') is not True or fresh.get('roots') != 128 or fresh.get('early_roots', 0) < 100
            or fresh.get('protected_gameplay_reads') is not False):
        raise PermissionError('fresh independent validation receipt required')
    directory = Path(output).parent / 'live-groups'
    artifacts = [item for item in old['group_artifacts'] if item['split'] == 'train']
    new_live = set()
    for position, reference in zip(plan['positions'], execution['parent_receipts'], strict=True):
        receipt = c.read(c.verify(reference))
        if not receipt['success']:
            if not receipt['unsupported']:
                raise ValueError('unexpected teacher failure retained')
            continue
        native = json.loads(data.bound(prior.legacy_reference(receipt['output'])).read_text())
        group = prior.normalized_label(native, position, plan)
        new_live.add(group['root_group_id'])
        payload = (json.dumps(group, sort_keys=True, separators=(',', ':')) + '\n').encode('ascii')
        path = directory / (position['position_id'] + '.json.gz')
        c.immutable(path, gzip.compress(payload, mtime=0))
        artifacts.append(dict(group_id=group['group_id'], root_group_id=group['root_group_id'],
                              split='train', edges=group['edges'], artifact=data.record(path)))
    train_roots = {item['root_group_id'] for item in artifacts}
    if len(train_roots) > 2048:
        raise ValueError('cumulative TRAIN cap exceeded')
    live_roots = merge_live_roots(old.get('live_root_ids', []), new_live, train_roots)
    train_features = set()
    for item in artifacts:
        with gzip.open(data.bound(item['artifact']), 'rt') as stream:
            group = json.load(stream)
        train_features.add(data.feature_key(group['parent_active']))
        train_features.update(data.feature_key(child['active']) for child in group['successors'])
    for item in fresh['group_artifacts']:
        with gzip.open(data.bound(item['artifact']), 'rt') as stream:
            group = json.load(stream)
        if any(data.feature_key(active) in train_features for active in
               [group['parent_active'], *[child['active'] for child in group['successors']]]):
            raise ValueError('TRAIN/reflected VALIDATION feature overlap')
    artifacts.extend(fresh['group_artifacts'])
    combined = combined_live_provenance(old.get('live_root_ids', []), new_live,
        c.read(c.verify(old['new_live_provenance'])), c.read(c.verify(plan['live_training_provenance'])))
    combined.update(retained_provenance=old['new_live_provenance'],
                    new_provenance=plan['live_training_provenance'])
    provenance = c.immutable(Path(output).parent / 'COMBINED_LIVE_PROVENANCE.json', combined)
    value = dict(old, group_artifacts=artifacts, live_root_ids=live_roots,
                 independent_comparable_early_validation_root_groups=fresh['early_roots'],
                 new_live_provenance=provenance, fresh_validation=fresh_validation,
                 outcomes_as_targets=False, production_training_admitted=True)
    return c.immutable(output, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(prepare(parser.parse_args().plan)))
