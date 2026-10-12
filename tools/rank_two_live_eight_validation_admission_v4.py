"""Check a frozen validation bank against cumulative eligible TRAIN features."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tools import rank_two_live_eight_data_v5 as live
from tools import rank_two_live_eight_validation_v6 as validation
from tools import rank_two_focused_work_v1 as work

c, data = live.c, live.data


def ready_inputs(plan, current):
    if current.get('pending_exposures') or current.get('active_claims'):
        raise PermissionError('close all exposures and unresolved claims before admission')
    for field, current_field in dict(
            declaration='live_eight_continuation_05_declaration',
            live_teacher_plan='live_eight_continuation_05_teacher_plan',
            validation_teacher_plan='live_eight_continuation_05_validation_teacher_plan',
            ancestry='live_eight_ancestry_validated').items():
        if not plan.get(field) or plan[field] != current.get(current_field):
            raise PermissionError('current continuation05 input binding differs: ' + field)
    declaration = c.read(c.verify(plan['declaration']))
    validation.verify_recipe(declaration)
    if plan['retained_train_corpus'] != declaration['retained_train_corpus']:
        raise PermissionError('retained TRAIN corpus differs from the frozen declaration')
    union = c.read(c.verify(plan['ancestry']))
    if not union.get('passed') or union['layers'] != current.get('live_eight_census_layers'):
        raise PermissionError('full current prior and every incremental ancestry layer required')


def train_feature_keys(retained, teacher_plan, execution, boundary=lambda: None):
    if not execution.get('passed'):
        raise ValueError('complete live TRAIN teacher execution required')
    features, roots, new_live = set(), set(), set()
    groups = 0

    def add(group, expected_root):
        nonlocal groups
        if group['split'] != 'train' or group['root_group_id'] != expected_root:
            raise PermissionError('held-out or changed root cannot enter TRAIN')
        roots.add(expected_root)
        features.add(data.feature_key(group['parent_active']))
        features.update(data.feature_key(child['active']) for child in group['successors'])
        groups += 1

    for item in retained['group_artifacts']:
        if item['split'] != 'train':
            continue
        boundary()
        with gzip.open(data.bound(item['artifact']), 'rt') as stream:
            add(json.load(stream), item['root_group_id'])
    for position, reference in zip(teacher_plan['positions'], execution['parent_receipts'], strict=True):
        boundary()
        receipt = c.read(c.verify(reference))
        if receipt['position_id'] != position['position_id']:
            raise ValueError('teacher parent identity differs')
        if not receipt['success']:
            if not receipt['unsupported']:
                raise ValueError('unexpected live TRAIN label failure retained')
            continue
        native = json.loads(data.bound(live.prior.legacy_reference(receipt['output'])).read_text())
        group = live.prior.normalized_label(native, position, teacher_plan)
        add(group, position['root_group_id'])
        new_live.add(group['root_group_id'])
    live_roots = live.merge_live_roots(retained.get('live_root_ids', []), new_live, roots)
    if len(roots) > 2048:
        raise ValueError('cumulative TRAIN root cap exceeded')
    return features, dict(train_roots=len(roots), live_roots=len(live_roots),
        retained_roots=len(roots - set(live_roots)), train_groups=groups,
        train_feature_keys=len(features), eligible_root_ids=sorted(roots),
        live_root_ids=live_roots, new_live_root_ids=sorted(new_live),
        all_historical_validation_excluded=True, outcomes_as_targets=False)


def admit(plan_path):
    plan = c.read(plan_path)
    current = work.checked(plan, 'train', launch=True)
    ready_inputs(plan, current)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__):
        raise ValueError('validation admission producer changed')
    for reference in plan['inputs'].values():
        c.verify(reference)
    retained = c.read(c.verify(plan['retained_train_corpus']))
    live_plan = c.read(c.verify(plan['live_teacher_plan']))
    if live_plan.get('inputs', {}).get('declaration') != plan['declaration']:
        raise PermissionError('live TRAIN teacher belongs to a different declaration')
    execution = c.read(c.verify(plan['live_teacher_execution']))
    if execution['plan'] != plan['live_teacher_plan']:
        raise ValueError('live TRAIN teacher binding differs')
    live.prior.verify_teacher_plan(live_plan)
    validation.verify_teacher_plan(c.read(c.verify(plan['validation_teacher_plan'])))
    features, admission = train_feature_keys(retained, live_plan, execution,
        lambda: work.checked(plan, 'train'))
    out = Path(plan['output'])
    train_ref = c.immutable(out / 'TRAIN_FEATURE_ADMISSION.json', dict(passed=True, **admission))
    fresh = validation.aggregate(c.verify(plan['validation_teacher_plan']),
        c.verify(plan['validation_teacher_execution']), features, out / 'validation')
    evidence = c.read(c.verify(fresh))
    result = dict(passed=True, plan=c.record(plan_path), fresh_validation=fresh,
        train_feature_admission=train_ref, validation_roots=evidence['roots'],
        comparable_early_roots=evidence['early_roots'],
        exclusions=evidence['exclusions'], training_eligible=False,
        outcomes_as_targets=False, protected_gameplay_reads=False,
        requires_spent_exposure_closure_before_training=True)
    c.immutable(out / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(admit(parser.parse_args().plan)))
