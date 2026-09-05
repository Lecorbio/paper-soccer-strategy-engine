#!/usr/bin/env python3
"""Freeze unprotected attribution after exactly two completed trained failures.

This diagnostic bridge recommends one approved intervention category. It never
starts attempt three, changes a training recipe, or grants advancement authority.
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


def profile_menu():
    maintained = pilot.selection
    return {'qat_and_scales': {name: trainer.qat_profile_contract(name) for name in trainer.QAT_PROFILES},
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


def body(root):
    previous = completed_pair(root)
    evidence = []
    for row in previous:
        phases = [phase_evidence(row['pilot'])] if row.get('stage') == 'full' else []
        phases.append(phase_evidence(row))
        terminal_path = Path(row['context']) / row['phase'] / ('attempt-outcome.json' if row.get('stage') == 'full' else 'pilot-outcome.json')
        terminal = bound_document(row['outcome'], terminal_path)
        stage = row['rejection_stage'] if row.get('stage') == 'full' else terminal['status']
        evidence.append({'attempt': row['attempt'], 'outcome': row['outcome'], 'rejection_stage': stage,
                         'completed_attempt_count': 1, 'phases': phases, 'downstream': downstream_evidence(row)})
    paths = {
        'driver': Path(__file__), 'attempt_validator': Path(attempts.__file__),
        'trainer': Path(trainer.__file__), 'selection': Path(pilot.selection.__file__),
        'pilot_selection': Path(pilot.__file__), 'teacher_profiles': Path(pilot.selection.pipeline.__file__),
        'campaign': Path(campaign.__file__),
    }
    for name in ('full_outcome', 'full_selection', 'search', 'category_profile',
                 'opponent_suite', 'development', 'timing_instrumentation'):
        paths[name] = campaign.REPO / 'tools' / ('compact_value_bfm_' + name + '_v2.py')
    producers = {name: campaign.record(path) for name, path in paths.items()}
    return {'schema': campaign.ID + '.attribution.v2', 'policy': POLICY, 'producers': producers,
            'completed_unsuccessful_trained_attempts': 2, 'attempts': evidence,
            'maintained_profile_menu': profile_menu(), 'recommendation': recommendation(evidence),
            'protected_results_used': False, 'live_results_used': False,
            'new_training_started': False, 'qualification_passed': False, 'campaign_success': False}


def produce(root):
    root = Path(root).resolve()
    if (root / 'attribution/after-two-attempts.json').exists():
        return validate(root)
    return campaign.seal(root / 'attribution/after-two-attempts.json', body(root))


def validate(root):
    root = Path(root).resolve()
    document = campaign.read(root / 'attribution/after-two-attempts.json')
    expected = body(root)
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
    parser.add_argument('command', choices=('record', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root):
        result = produce(args.root) if args.command == 'record' else validate(args.root)
    print(json.dumps(result['recommendation']), flush=True)


if __name__ == '__main__':
    main()
