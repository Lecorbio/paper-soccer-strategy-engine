#!/usr/bin/env python3
"""Freeze the historical after-two or concrete after-three failure attribution.

This diagnostic bridge recommends one approved intervention category. It never
starts training, changes a training recipe, or grants advancement authority.
The maintained attempt validators reproduce outcomes and their source bindings;
only explicitly projected unprotected metrics enter the attribution decision.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_attempt_v2 as attempts
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_pilot_selection_v2 as pilot
from tools import compact_value_bfm_timing_instrumentation_v2 as instrumentation

trainer = pilot.trainer
POLICY = {
    'schema': campaign.ID + '.attribution-policy.v2',
    'completed_attempts': [1, 2], 'pilot_and_full_count_together': True,
    'smoke_counts': False, 'evidence_scope': 'unprotected-only',
    'decision_inputs': 'terminal-phase-selected-ranking-arm-seeds;all-other-seeds-diagnostic',
    'priority': ['quantization-retention-loss-or-excess-action-flips', 'float-quality-or-covered-ranking-deficit',
                 'completed-search-or-strength-rejection'],
    'thresholds': 'existing-canonical-retention-and-pilot-ranking-policy-only',
    'interpretation': 'diagnostic-next-experiment-recommendation;not-causal-proof',
    'unknown_fields_followed': False, 'protected_metrics_used': False,
    'live_metrics_used': False, 'transcripts_in_output': False,
    'attempt_three_authorized_by_receipt': False,
}
RETENTION_PROFILE = 'retention-first-low-rate-v1'
AFTER_THREE_POLICY = {
    **POLICY, 'schema': campaign.ID + '.after-three-attribution-policy.v2',
    'completed_attempts': [1, 2, 3],
    'decision_inputs': 'all-nine-completed-third-pilot-retention-verdicts;prior-attempts-context-only',
    'priority': ['explicit-retention-first-QAT-scales-hypothesis'],
    'third_pilot_required_status': 'offline-rejected',
    'third_pilot_all_nine_retention_failed': True,
    'attempt_four_authorized_by_receipt': False,
}
AFTER_THREE_RECOMMENDATION = {
    'category': 'qat-and-scales', 'existing_profile_to_consider': RETENTION_PROFILE,
    'selected_execution_profile': None, 'attempt_four_may_start': False,
    'causal_attribution_proven': False, 'status': 'awaiting-source-bound-profile-integration',
    'hypothesis': 'retention-first scale and epoch selection against frozen pre-QAT float validation, with quarter QAT learning rate',
    'basis': 'all-nine-completed-refined-third-pilot-retention-failures',
    'limits': ['This is the selected QAT/scales hypothesis, not a proven cause or an admission result.',
               'Fresh training, accumulated exclusions and every original gate remain required.'],
}


def path(root, *, after_attempts=2):
    if type(after_attempts) is not int or after_attempts not in (2, 3):
        raise ValueError('attribution supports only the frozen after-two or after-three route')
    name = 'two' if after_attempts == 2 else 'three'
    return Path(root).resolve() / f'attribution/after-{name}-attempts.json'


def number(value, *, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (minimum is not None and value < minimum) or (maximum is not None and value > maximum)):
        raise ValueError('invalid unprotected attribution metric')
    return value


def count(value):
    if type(value) is not int or value < 0:
        raise ValueError('invalid unprotected attribution count')
    return value


def boolean(value):
    if type(value) is not bool:
        raise ValueError('invalid unprotected attribution verdict')
    return value


def bound_document(record, expected):
    """Reject a redirected source before hashing or opening the target."""
    expected = Path(expected)
    if (Path(record['path']).absolute() != expected.absolute()
            or expected.resolve() != expected.absolute()):
        raise ValueError('attribution source leaves its fixed unprotected stage path')
    campaign.verify(record)
    return campaign.read(expected)


def ranking(metrics):
    groups = count(metrics['groups']); comparable = count(metrics['comparable_groups'])
    if comparable > groups:
        raise ValueError('ranking comparable count exceeds total groups')
    result = {'groups': groups, 'comparable_groups': comparable}
    if groups == 0:
        if comparable:
            raise ValueError('empty ranking stratum has comparable groups')
        return {**result, 'coverage_passed': False, 'mean_teacher_regret': None,
                'top1_agreement': None, 'float_vs_quantized_action_flip_rate': None}
    result.update({key: number(metrics[key], minimum=0, maximum=1 if key != 'mean_teacher_regret' else None)
                   for key in ('mean_teacher_regret', 'top1_agreement', 'float_vs_quantized_action_flip_rate')})
    return {**result, 'coverage_passed': pilot.coverage(result)}


def seed_metrics(row):
    receipt = row['seed_receipt']
    reports = {}
    for frame in ('float_validation', 'quantized_validation'):
        source = receipt[frame]
        reports[frame] = {name: {
            'sign_accuracy': number(source[name]['sign_accuracy'], minimum=0, maximum=1),
            'weighted_huber': number(source[name]['weighted_huber'], minimum=0),
        } for name in ('common_adjudicator', 'canonical_validation')}
        reports[frame]['successor_ranking'] = ranking(source['successor_ranking'])
    effects = {}
    for name, sign, huber in (
        ('common_adjudicator', trainer.COMMON_MINIMUM_SIGN, trainer.COMMON_MAXIMUM_HUBER),
        ('canonical_validation', trainer.CANONICAL_MINIMUM_SIGN, trainer.CANONICAL_MAXIMUM_HUBER),
    ):
        floating, quantized = (reports[frame][name] for frame in ('float_validation', 'quantized_validation'))
        float_pass = floating['sign_accuracy'] >= sign and floating['weighted_huber'] <= huber
        quantized_pass = quantized['sign_accuracy'] >= sign and quantized['weighted_huber'] <= huber
        sign_loss = floating['sign_accuracy'] - quantized['sign_accuracy']
        relative_failure = (sign_loss >= trainer.MAXIMUM_SIGN_LOSS
                            or quantized['weighted_huber'] > floating['weighted_huber'] * trainer.MAXIMUM_HUBER_RATIO)
        effects[name] = {'float_absolute_passed': float_pass, 'quantized_absolute_passed': quantized_pass,
                        'sign_accuracy_loss': sign_loss,
                        'huber_increase': quantized['weighted_huber'] - floating['weighted_huber'],
                        'relative_retention_failed': relative_failure,
                        'quantization_boundary_crossing': float_pass and not quantized_pass}
    floating, quantized = (reports[frame]['successor_ranking'] for frame in ('float_validation', 'quantized_validation'))
    if not floating['groups'] or not quantized['groups']:
        raise ValueError('completed seed attribution requires nonempty overall ranking evidence')
    if (floating['groups'], floating['comparable_groups']) != (quantized['groups'], quantized['comparable_groups']):
        raise ValueError('float and quantized ranking reports use different groups')
    effects['ranking'] = {'regret_increase': quantized['mean_teacher_regret'] - floating['mean_teacher_regret'],
                          'action_flip_rate': quantized['float_vs_quantized_action_flip_rate'],
                          'coverage_passed': floating['coverage_passed'] and quantized['coverage_passed']}
    return {'lambda': row['weight'], 'seed': row['seed'], 'reports': reports, 'quantization_effects': effects,
            'canonical_retention_passed': boolean(receipt['offline_gate']['passed'])}


def phase_evidence(previous):
    context, phase = Path(previous['context']), previous['phase']
    directory = context / phase
    training = bound_document(previous['training'], directory / 'training.json')
    full = previous.get('stage') == 'full'
    selected = bound_document(previous['selection'], directory / ('full-model-selection.json' if full else 'model-selection.json'))
    if training.get('smoke') is not False or training.get('mandatory_training_verified') is not True:
        raise ValueError('attribution requires completed nonsmoke teacher training')
    seeds = [seed_metrics(row) for row in training['results']]
    index = {(row['lambda'], row['seed']): row for row in seeds}
    if len(index) != len(seeds):
        raise ValueError('duplicate training seed in attribution')
    arms = []
    for arm in selected['arms']:
        key = (arm['lambda'], arm['seed'])
        if key not in index:
            raise ValueError('selected attribution arm lacks a completed seed')
        arms.append({'lambda': key[0], 'seed': key[1],
                     'overall': ranking(arm['overall']), 'early': ranking(arm['early']),
                     'canonical_retention_passed': boolean(arm['canonical_retention_passed']),
                     'source_reserve': count(arm['source_reserve'])})
    control = next((arm for arm in arms if arm['lambda'] == 0), None)
    if control is None:
        raise ValueError('attribution lacks its trained scalar control')
    comparisons = []
    for arm in arms:
        if arm['lambda'] == 0:
            continue
        strata = {}
        for name in ('overall', 'early'):
            left, right = control[name], arm[name]
            covered = left['coverage_passed'] and right['coverage_passed']
            reduction = pilot.selection._regret_reduction(left['mean_teacher_regret'], right['mean_teacher_regret']) if covered else None
            flip = right['float_vs_quantized_action_flip_rate'] - left['float_vs_quantized_action_flip_rate'] if covered else None
            strata[name] = {'coverage_passed': covered, 'regret_reduction': reduction,
                            'flip_rate_increase': flip,
                            'ranking_deficit': covered and reduction < pilot.SELECTION_POLICY['minimum_regret_reduction'],
                            'excess_flip_increase': covered and flip > pilot.SELECTION_POLICY['maximum_flip_increase']}
        comparisons.append({'lambda': arm['lambda'], 'seed': arm['seed'], 'strata': strata})
    return {'phase': phase, 'training': previous['training'], 'selection': previous['selection'],
            'seeds': seeds, 'arms': arms, 'comparisons': comparisons}


def downstream_evidence(previous):
    """Project summaries already reproduced by failed_full; never follow arbitrary fields."""
    result = {'search': None, 'suites': [], 'development': None, 'pilot_screen': None}
    pilot_evidence = previous['pilot'] if previous.get('stage') == 'full' else previous
    if pilot_evidence['screen'] is not None:
        raw = pilot_evidence['screen'][1]['result']
        result['pilot_screen'] = {key: count(raw[key]) for key in ('games', 'candidate_wins', 'failures')}
    if previous.get('stage') != 'full':
        return result
    source = previous['source_selection']
    if source is not None:
        shares = {name: {key: number(row['shares'][key], minimum=0, maximum=1)
                         for key in instrumentation.CATEGORIES}
                  for name, row in source['category_profile']['variants'].items()}
        result['search'] = {
            'receipt': previous['stages']['search'], 'retained_variants': list(source['retained_variants']),
            'timing': {name: {key: boolean(row[key]) if key == 'passed' else number(row[key])
                             for key in ('throughput_gain', 'p95_regression', 'passed')}
                       for name, row in source['throughput_and_latency'].items()},
            'baseline_wins': count(source['clocked_strength']['baseline_wins']),
            'paired_win_deltas': {name: number(delta) for name, delta in source['clocked_strength']['paired_win_deltas'].items()},
            'category_shares': shares, 'category_shares_authorize_speed_retention': False,
        }
    for stage, checked in zip(('screen', 'confirmation'), previous['suites']):
        report = checked[0]
        result['suites'].append({'stage': stage, 'receipt': previous['stages'][stage],
            'passed': boolean(report['passed']), 'equal_weight_improvement': number(report['equal_weight_improvement']),
            'paired_95_interval': [number(value) for value in report['paired_95_interval']],
            'failures': len(report['failures']),
            'opponents': {name: {key: count(row[key]) if key == 'root_pairs' else number(row[key])
                                for key in ('root_pairs', 'candidate_win_rate', 'control_win_rate', 'improvement')}
                          for name, row in report['opponents'].items()}})
    if previous['development'] is not None:
        report = previous['development']
        result['development'] = {'receipt': previous['stages']['development'], 'passed': boolean(report['passed']),
            **{key: count(report[key]) for key in ('games', 'candidate_wins', 'failures')},
            'wins_by_color': [count(value) for value in report['candidate_wins_by_color']],
            'paired_lower_95': number(report['paired_lower_95'], minimum=0, maximum=1)}
    return result


def recommendation(evidence):
    signals = {'qat-and-scales': [], 'harder-teacher-ranking': [], 'one-search-intervention': []}
    gaps = []
    for attempt in evidence:
        terminal = attempt['phases'][-1]
        index = {(row['lambda'], row['seed']): row for row in terminal['seeds']}
        for arm in terminal['arms']:
            if arm['lambda'] == 0:
                continue
            key = {'attempt': attempt['attempt'], 'phase': terminal['phase'], 'lambda': arm['lambda'], 'seed': arm['seed']}
            effects = index[arm['lambda'], arm['seed']]['quantization_effects']
            for name in ('common_adjudicator', 'canonical_validation'):
                effect = effects[name]
                if effect['relative_retention_failed'] or effect['quantization_boundary_crossing']:
                    signals['qat-and-scales'].append({**key, 'stratum': name, 'reason': 'float-to-quantized-retention-loss'})
                if not effect['float_absolute_passed']:
                    signals['harder-teacher-ranking'].append({**key, 'stratum': name, 'reason': 'float-already-below-existing-absolute-floor'})
        for comparison in terminal['comparisons']:
            key = {'attempt': attempt['attempt'], 'phase': terminal['phase'], 'lambda': comparison['lambda'], 'seed': comparison['seed']}
            for name, stratum in comparison['strata'].items():
                if not stratum['coverage_passed']:
                    gaps.append({**key, 'stratum': name, 'reason': 'insufficient-comparable-groups'})
                elif stratum['excess_flip_increase']:
                    signals['qat-and-scales'].append({**key, 'stratum': name, 'reason': 'flip-increase-exceeds-frozen-pilot-limit'})
                if stratum['ranking_deficit']:
                    signals['harder-teacher-ranking'].append({**key, 'stratum': name, 'reason': 'covered-regret-reduction-below-frozen-pilot-target'})
        if attempt['rejection_stage'] in ('search', 'screen', 'confirmation', 'development'):
            signals['one-search-intervention'].append({'attempt': attempt['attempt'], 'reason': 'completed-unprotected-strength-rejection',
                                                       'stage': attempt['rejection_stage']})
    category = next((name for name, rows in signals.items() if rows), None)
    return {'category': category, 'signals': signals, 'coverage_gaps': gaps,
            'rule': POLICY['priority'], 'causal_attribution_proven': False,
            'existing_profile_to_consider': trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE if category == 'qat-and-scales' else None,
            'selected_execution_profile': None, 'attempt_three_may_start': False,
            'status': 'awaiting-source-bound-profile-integration' if category else 'insufficient-unprotected-attribution-evidence',
            'limitations': [
                'Observed diagnostics select an experiment category; they do not establish the cause of lost strength.',
                'Existing hardest-5pct-2m-v1 changes the frozen deep-label budget and density; this bridge does not apply it.',
                'Category timing alone cannot select cache, widening, or within-search reuse; one profile needs independent invariants and strength evidence.',
                'A new attempt still requires fresh accepted-teacher training, accumulated exclusions, and every original admission gate.',
            ]}


def profile_menu(*, after_attempts=2):
    path(Path('.'), after_attempts=after_attempts)
    maintained = pilot.selection
    # Historical after-two receipts froze exactly this menu. New registrations
    # must not change their substantive validation when read by a newer source.
    profiles = (trainer.STANDARD_QAT_PROFILE, trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE)
    if after_attempts == 3:
        profiles += (RETENTION_PROFILE,)
    return {'qat_and_scales': {name: trainer.qat_profile_contract(name) for name in profiles},
            'teacher_ranking': {name: maintained.pipeline.teacher_ranking_policy(name)
                                for name in maintained.pipeline.TEACHER_RANKING_PROFILES},
            'single_search': sorted(attempts.rank4_gate_support.SEARCH_PROFILES - {'standard-v1'}),
            'combine_search_profiles': False, 'cross_turn_persistence': False}


def completed_pair(root):
    """Check both terminal slots before any validation or metric projection."""
    root = Path(root).resolve()
    for attempt in (1, 2):
        full = root / 'phases' / f'attempt-{attempt:03d}-full'
        phase = full.name if full.exists() else f'attempt-{attempt:03d}-pilot'
        context = root / 'phases' / phase
        terminal = context / phase / ('attempt-outcome.json' if full.exists() else 'pilot-outcome.json')
        if terminal.resolve() != terminal.absolute() or not terminal.is_file():
            raise ValueError('two completed unsuccessful trained attempts are required before attribution')
    previous = [attempts.failed_attempt(root, attempt) for attempt in (1, 2)]
    if [row['attempt'] for row in previous] != [1, 2]:
        raise ValueError('attribution requires two distinct verified attempt identities')
    return previous


def completed_three(root):
    """Reject incomplete/protected branches before invoking outcome validators."""
    root = Path(root).resolve()
    terminals = []
    for attempt in (1, 2, 3):
        full = root / 'phases' / f'attempt-{attempt:03d}-full'
        if attempt == 3 and full.exists():
            raise ValueError('after-three retention attribution requires an offline-rejected third pilot')
        phase = full.name if full.exists() else f'attempt-{attempt:03d}-pilot'
        terminal = root / 'phases' / phase / phase / ('attempt-outcome.json' if full.exists() else 'pilot-outcome.json')
        if terminal.resolve() != terminal.absolute() or not terminal.is_file():
            raise ValueError('three completed unsuccessful trained attempts are required before attribution')
        terminals.append((attempt, terminal, full.exists()))
    for attempt, terminal, full in terminals:
        document = campaign.read(terminal)
        if full:
            # The terminal-outcome dispatcher may open protected/live evidence;
            # this route accepts only the maintained unprotected full outcome.
            if (document.get('schema') != campaign.ID + '.full-attempt-outcome.v2'
                    or document.get('status') != 'completed-unsuccessful'
                    or document.get('rejection_stage') not in ('full-offline', 'search', 'screen', 'confirmation', 'development')):
                raise ValueError('after-three attribution accepts only completed unprotected failures')
        elif (document.get('schema') != campaign.ID + '.pilot-outcome.v2'
                or document.get('admitted') is not False or document.get('campaign_success') is not False
                or document.get('status') not in ('offline-rejected', 'rank4-screen-rejected')
                or attempt == 3 and document.get('status') != 'offline-rejected'):
            raise ValueError('after-three attribution requires completed unprotected failed pilots')
    previous = [attempts.failed_attempt(root, attempt) for attempt in (1, 2, 3)]
    if [row['attempt'] for row in previous] != [1, 2, 3] or any(row.get('terminal_outcome') for row in previous):
        raise ValueError('after-three attribution lost its distinct unprotected attempt identities')
    for row, (_attempt, terminal, full) in zip(previous, terminals):
        if (Path(row['context']).resolve() != terminal.parent.parent
                or row['phase'] != terminal.parent.name or (row.get('stage') == 'full') is not full):
            raise ValueError('after-three validator returned a redirected attempt context')
    return previous


def third_pilot_retention(root, bindings):
    """Reproduce the narrow new-route facts from bound metadata, without forwards."""
    root = Path(root).resolve()
    context = root / 'phases/attempt-003-pilot'; directory = context / context.name
    if (root / 'phases/attempt-003-full').exists() or (directory / 'rank4-screen').exists():
        raise ValueError('retention hypothesis cannot abandon a full or spent screen branch')
    contract = bound_document(bindings['contract'], context / 'campaign.json')
    outcome = bound_document(bindings['outcome'], directory / 'pilot-outcome.json')
    selection = bound_document(bindings['selection'], directory / 'model-selection.json')
    training = bound_document(bindings['training'], directory / 'training.json')
    refined = trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE
    if (contract.get('attempt') != 3 or contract.get('phase') != 'pilot'
            or contract.get('qat_profile') != refined
            or contract.get('qat_profile_contract') != trainer.qat_profile_contract(refined)
            or outcome.get('schema') != campaign.ID + '.pilot-outcome.v2'
            or outcome.get('status') != 'offline-rejected' or outcome.get('admitted') is not False
            or outcome.get('campaign_success') is not False or outcome.get('selection') != bindings['selection']
            or selection.get('schema') != campaign.ID + '.pilot-model-selection.v2'
            or selection.get('status') != 'offline-rejected-before-rank4-screen'
            or selection.get('selected') is not None or selection.get('pilot_admitted') is not False
            or selection.get('campaign_success') is not False or selection.get('training') != bindings['training']
            or training.get('schema') != campaign.ID + '.training.v2' or training.get('smoke') is not False
            or training.get('mandatory_training_verified') is not True):
        raise ValueError('third-pilot metadata does not bind the completed refined retention failure')
    rows = training['results']
    expected = {(weight, seed) for weight in (0, .1, .25) for seed in trainer.FIXED_SEEDS}
    if (len(rows) != 9 or any(isinstance(row['weight'], bool) or type(row['seed']) is not int for row in rows)
            or {(row['weight'], row['seed']) for row in rows} != expected):
        raise ValueError('third-pilot retention attribution requires all nine distinct trained seeds')
    seeds = []
    for row in sorted(rows, key=lambda value: (value['weight'], value['seed'])):
        receipt = row['seed_receipt']; projected = seed_metrics(row)
        # Never pass unknown metric fields, protected references or transcripts
        # to the gate; its decision uses only these two allowlisted strata.
        frames = [{name: projected['reports'][frame][name]
                   for name in ('common_adjudicator', 'canonical_validation')}
                  for frame in ('float_validation', 'quantized_validation')]
        gate = trainer.offline_advancement_gate(*frames)
        if (receipt.get('qat_profile') != refined
                or receipt.get('qat_profile_contract') != trainer.qat_profile_contract(refined)
                or receipt.get('offline_gate') != gate or gate['passed'] is not False):
            raise ValueError('all nine third-pilot retention failures must reproduce their actual gates')
        seeds.append({**projected, 'offline_gate': gate})
    if ([arm['lambda'] for arm in selection['arms']] != [0, .1, .25]
            or any(arm.get('canonical_retention_passed') is not False for arm in selection['arms'])):
        raise ValueError('third-pilot selected arms disagree with all-nine retention rejection')
    return {'sources': {key: bindings[key] for key in ('contract', 'outcome', 'training', 'selection')},
            'attempt': 3, 'phase': context.name, 'prior_qat_profile': refined,
            'completed_seed_count': 9, 'retention_failed_seed_count': 9,
            'all_nine_retention_failed': True, 'status': 'offline-rejected', 'seeds': seeds}


def _project_attempts(previous):
    evidence = []
    for row in previous:
        phases = [phase_evidence(row['pilot'])] if row.get('stage') == 'full' else []
        phases.append(phase_evidence(row))
        terminal_path = Path(row['context']) / row['phase'] / ('attempt-outcome.json' if row.get('stage') == 'full' else 'pilot-outcome.json')
        terminal = bound_document(row['outcome'], terminal_path)
        stage = row['rejection_stage'] if row.get('stage') == 'full' else terminal['status']
        evidence.append({'attempt': row['attempt'], 'outcome': row['outcome'], 'rejection_stage': stage,
                         'completed_attempt_count': 1, 'phases': phases, 'downstream': downstream_evidence(row)})
    return evidence


def _producers():
    paths = {
        'driver': Path(__file__), 'attempt_validator': Path(attempts.__file__),
        'trainer': Path(trainer.__file__), 'selection': Path(pilot.selection.__file__),
        'pilot_selection': Path(pilot.__file__), 'teacher_profiles': Path(pilot.selection.pipeline.__file__),
        'campaign': Path(campaign.__file__),
    }
    for name in ('full_outcome', 'full_selection', 'search', 'category_profile',
                 'opponent_suite', 'development', 'timing_instrumentation'):
        paths[name] = campaign.REPO / 'tools' / ('compact_value_bfm_' + name + '_v2.py')
    return {name: campaign.record(path) for name, path in paths.items()}


def body(root):
    evidence = _project_attempts(completed_pair(root))
    return {'schema': campaign.ID + '.attribution.v2', 'policy': POLICY, 'producers': _producers(),
            'completed_unsuccessful_trained_attempts': 2, 'attempts': evidence,
            'maintained_profile_menu': profile_menu(), 'recommendation': recommendation(evidence),
            'protected_results_used': False, 'live_results_used': False,
            'new_training_started': False, 'qualification_passed': False, 'campaign_success': False}


def body_after_three(root):
    root = Path(root).resolve()
    previous = completed_three(root)
    third = previous[-1]
    bindings = {key: third[key] for key in ('outcome', 'training', 'selection')}
    bindings['contract'] = campaign.record(Path(third['context']) / 'campaign.json')
    facts = third_pilot_retention(root, bindings)
    evidence = _project_attempts(previous)
    return {'schema': campaign.ID + '.after-three-attribution.v2',
            'policy': AFTER_THREE_POLICY, 'producers': _producers(),
            'completed_unsuccessful_trained_attempts': 3, 'attempts': evidence,
            'third_pilot_retention': facts,
            'maintained_profile_menu': profile_menu(after_attempts=3),
            'recommendation': AFTER_THREE_RECOMMENDATION,
            'protected_results_used': False, 'live_results_used': False,
            'new_training_started': False, 'qualification_passed': False, 'campaign_success': False}


def produce(root, *, after_attempts=2):
    root = Path(root).resolve()
    output = path(root, after_attempts=after_attempts)
    if output.exists():
        return validate(root, after_attempts=after_attempts)
    return campaign.seal(output, body(root) if after_attempts == 2 else body_after_three(root))


def validate(root, *, after_attempts=2):
    root = Path(root).resolve()
    document = campaign.read(path(root, after_attempts=after_attempts))
    expected = body(root) if after_attempts == 2 else body_after_three(root)
    if set(document['producers']) != set(expected['producers']):
        raise ValueError('attribution source closure is incomplete')
    for record in document['producers'].values():
        campaign.verify(record)
    # Reproduce all decisions with current validators while retaining the actual
    # historical producer paths from the immutable execution snapshot.
    excluded = {'body_sha256', 'producers'}
    if ({key: value for key, value in document.items() if key not in excluded}
            != {key: value for key, value in expected.items() if key not in excluded}):
        raise ValueError('attribution differs from verified unprotected attempt evidence')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--after-attempts', type=int, choices=(2, 3), default=2)
    parser.add_argument('command', choices=('record', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root):
        result = (produce(args.root, after_attempts=args.after_attempts) if args.command == 'record'
                  else validate(args.root, after_attempts=args.after_attempts))
    print(json.dumps(result['recommendation']), flush=True)


if __name__ == '__main__':
    main()
