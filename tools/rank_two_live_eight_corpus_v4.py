"""Assemble a strict complete bank without inheriting a prior bank's quarantine gate."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_data_v4 as live
from tools import rank_two_focused_work_v1 as work


def ready_inputs(plan, current, admitted_plan, admission, train, fresh):
    if current.get('pending_exposures') or current.get('active_claims'):
        raise PermissionError('close all spent exposures and unresolved claims before corpus admission')
    for key in ('retained_train_corpus', 'live_teacher_plan', 'live_teacher_execution',
                'validation_teacher_plan', 'validation_teacher_execution'):
        if plan[key] != admitted_plan[key]:
            raise ValueError('cumulative admission input differs: ' + key)
    if (not admission.get('passed') or admission['plan'] != plan['validation_admission_plan']
            or not train.get('passed') or train.get('outcomes_as_targets') is not False
            or not train.get('all_historical_validation_excluded')):
        raise PermissionError('complete cumulative TRAIN feature admission required')
    if (not fresh.get('passed') or fresh.get('roots') != 128 or fresh.get('early_roots', 0) < 100
            or fresh.get('training_eligible') is not False
            or fresh.get('outcomes_as_targets') is not False
            or fresh.get('protected_gameplay_reads') is not False
            or fresh['plan'] != plan['validation_teacher_plan']
            or fresh['execution'] != plan['validation_teacher_execution']):
        raise PermissionError('frozen independent validation admission required')
    if (train['train_roots'] > 2048 or train['train_roots'] != len(set(train['eligible_root_ids']))
            or train['live_roots'] != len(set(train['live_root_ids']))
            or not set(train['live_root_ids']) <= set(train['eligible_root_ids'])):
        raise ValueError('cumulative root namespace or cap differs')
    if fresh.get('gate_approval') or fresh.get('excluded_root_ids') or fresh.get('exclusions'):
        raise PermissionError('strict bank requires all roots and no prior quarantine exception')
    roots = {row['root_group_id'] for row in fresh['group_artifacts']}
    if len(roots) != 128 or any(row['split'] != 'validation' for row in fresh['group_artifacts']):
        raise PermissionError('all128 actual validation roots required')


def strict_manifest(corpus, train_roots, fresh):
    result = dict(corpus)
    previous_gate = result.pop('validation_gate_approval', None)
    previous_accounting = result.get('validation_root_accounting')
    if previous_gate:
        result['prior_validation_gate_retained'] = previous_gate
    if previous_accounting:
        result['prior_validation_root_accounting_retained'] = previous_accounting
    result.update(train_root_group_ids=sorted(train_roots),
        eligible_validation_root_group_ids=sorted({row['root_group_id'] for row in fresh['group_artifacts']}),
        validation_root_accounting=dict(frozen=128, evaluated=128, excluded=0,
                                        comparable_early=fresh['early_roots']))
    return result


def assemble(plan_path):
    plan = c.read(plan_path)
    current = work.checked(plan, 'train', launch=True)
    c.resource_guard(plan['resource_budget'])
    if plan['producer'] != c.record(__file__):
        raise ValueError('corpus producer changed')
    for reference in plan['inputs'].values():
        c.verify(reference)
    union = c.read(c.verify(current['live_eight_ancestry_validated']))
    if not union['passed'] or union['layers'] != current['live_eight_census_layers']:
        raise PermissionError('full prior and all spent incremental ancestry layers required')
    admission = c.read(c.verify(plan['validation_admission']))
    admitted_plan = c.read(c.verify(plan['validation_admission_plan']))
    train = c.read(c.verify(admission['train_feature_admission']))
    fresh = c.read(c.verify(admission['fresh_validation']))
    ready_inputs(plan, current, admitted_plan, admission, train, fresh)
    retained = c.read(c.verify(plan['retained_train_corpus']))
    teacher_plan = c.read(c.verify(plan['live_teacher_plan']))
    execution = c.read(c.verify(plan['live_teacher_execution']))
    live.prior.verify_teacher_plan(teacher_plan)
    if not execution['passed'] or execution['plan'] != plan['live_teacher_plan']:
        raise ValueError('completed live TRAIN teacher binding differs')
    out = Path(plan['output'])
    artifacts = [item for item in retained['group_artifacts'] if item['split'] == 'train']
    new_live = set()
    for position, reference in zip(teacher_plan['positions'], execution['parent_receipts'], strict=True):
        work.checked(plan, 'train')
        receipt = c.read(c.verify(reference))
        if receipt['position_id'] != position['position_id']:
            raise ValueError('teacher parent identity changed')
        if not receipt['success']:
            if not receipt['unsupported']:
                raise ValueError('unexpected live teacher failure retained')
            continue
        native = json.loads(live.data.bound(live.prior.legacy_reference(receipt['output'])).read_text())
        group = live.prior.normalized_label(native, position, teacher_plan)
        if group['split'] != 'train' or group['root_group_id'] != position['root_group_id']:
            raise PermissionError('held-out or changed root cannot enter TRAIN')
        new_live.add(group['root_group_id'])
        path = out / 'live-groups' / (position['position_id'].replace(':', '-') + '.json.gz')
        c.immutable(path, gzip.compress(c.old.retained.canonical(group), mtime=0))
        artifacts.append(dict(group_id=group['group_id'], root_group_id=group['root_group_id'],
            split='train', edges=group['edges'], artifact=live.data.record(path)))
    roots = {item['root_group_id'] for item in artifacts}
    live_roots = live.merge_live_roots(retained.get('live_root_ids', []), new_live, roots)
    if (roots != set(train['eligible_root_ids']) or set(live_roots) != set(train['live_root_ids'])
            or new_live != set(train['new_live_root_ids']) or len(artifacts) != train['train_groups']):
        raise ValueError('assembled groups differ from the complete cumulative feature admission')
    combined = live.combined_live_provenance(retained.get('live_root_ids', []), new_live,
        c.read(c.verify(retained['new_live_provenance'])),
        c.read(c.verify(teacher_plan['live_training_provenance'])))
    combined.update(retained_provenance=retained['new_live_provenance'],
        new_provenance=teacher_plan['live_training_provenance'])
    provenance = c.immutable(out / 'COMBINED_LIVE_PROVENANCE.json', combined)
    fresh_ref = c.immutable(out / 'FRESH_VALIDATION.json', dict(fresh,
        ancestry=current['live_eight_ancestry_validated'], before_generation_validated=True,
        original_result=admission['fresh_validation'], cumulative_feature_admission=plan['validation_admission']))
    artifacts.extend(fresh['group_artifacts'])
    corpus = dict(retained, group_artifacts=artifacts, live_root_ids=live_roots,
        independent_comparable_early_validation_root_groups=fresh['early_roots'],
        new_live_provenance=provenance, fresh_validation=fresh_ref,
        outcomes_as_targets=False, production_training_admitted=True)
    corpus = strict_manifest(corpus, roots, fresh)
    corpus['retained_corpus'] = plan['retained_train_corpus']
    corpus_ref = c.immutable(out / 'CORPUS.json', corpus)
    result = dict(passed=True, corpus=corpus_ref, fresh_validation=fresh_ref,
        train_roots=len(roots), retained_roots=train['retained_roots'], live_roots=len(live_roots),
        validation_roots=128, early_roots=fresh['early_roots'],
        root_accounting=corpus['validation_root_accounting'],
        evaluated_validation_roots=128, excluded_validation_roots=0,
        training_mass=dict(retained=.5, live=.5), cumulative_feature_admission=plan['validation_admission'],
        feature_check_reused_with_identical_bound_inputs=True,
        protected_gameplay_reads=False, validation_never_used_for_updates=True)
    c.immutable(out / 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    print(json.dumps(assemble(parser.parse_args().plan)))
