#!/usr/bin/env python3
"""Close a genuinely completed unsuccessful full attempt using unprotected evidence.

This bridge runs no games or training. It reopens real seed references and the
maintained stage validators before recording a rejection. Incomplete stages,
operational interruptions, protected gates and live results cannot close an
attempt through this command.
"""
from __future__ import annotations

import argparse
import json
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
from tools import compact_value_bfm_development_v2 as development
from tools import compact_value_bfm_full_selection_v2 as full_selection
from tools import compact_value_bfm_opponent_suite_v2 as suite
from tools import compact_value_bfm_search_v2 as search


def validate_full_selection(root, context, phase):
    """Reproduce selection without creating a policy or replacing a receipt."""
    contract, parent = full_selection.validate_context(root, context, phase)
    directory = context / phase
    document = campaign.read(directory / 'full-model-selection.json')
    policy = attempts.bound(document['policy'], directory / 'full-selection-policy.json')
    if ({key: value for key, value in policy.items() if key not in ('body_sha256', 'source_closure')}
            != {**full_selection.policy(contract), 'context': campaign.record(context / 'campaign.json')}
            or not policy.get('source_closure')):
        raise ValueError('full outcome requires the frozen full-selection policy')
    for item in policy['source_closure']:
        campaign.verify(item)
    training_path = directory / 'training.json'
    training = attempts.bound(document['training'], training_path)
    weights = contract['full_training_roster']['lambdas']
    records = full_selection.validate_roster(training, weights)
    for key in ('datasets', 'source_routes'):
        if any(row['seed_receipt']['binding'][key] != records[0]['seed_receipt']['binding'][key] for row in records[1:]):
            raise ValueError('full outcome training seeds used different scalar corpora')
    maintained, trainer = full_selection.selection, full_selection.trainer
    for weight in weights:
        maintained._validate_seed_roster(
            [row['seed_receipt'] for row in sorted(records, key=lambda row: row['seed']) if row['weight'] == weight],
            ranking_weight=weight, qat_profile=full_selection.intervention.expected_qat_profile(contract),
            teacher_ranking_profile=maintained.pipeline.STANDARD_TEACHER_RANKING_PROFILE)
    campaign.verify(training['producer'])
    bundle = trainer.FrozenBundle.load(campaign.verify(contract['bundle']))
    audit, rankings = full_selection.validate_inputs(context, phase, contract, training, bundle)
    initial = trainer.load_float_checkpoint(campaign.verify(contract['inputs']['attempt_one_initial_checkpoint']),
                                            trainer.ARCHITECTURES['capacity-12x8'])
    evidence = {(row['weight'], row['seed']): full_selection.validate_seed(
        context, phase, row, contract, audit, rankings, bundle, initial) for row in records}
    scalar, candidate = full_selection.select_arms(records, evidence, rankings)
    selected = candidate if candidate['eligible_for_multi_opponent'] else None
    control = {'source': parent['inputs']['discrete_v3_deployment.cpp'],
               'runtime': parent['inputs']['attempt_zero_runtime'], 'role': 'frozen-deployed-strength-control'}
    for key in ('source', 'runtime'):
        campaign.verify(control[key])
    expected = {
        'schema': campaign.ID + '.full-model-selection.v2', 'policy': document['policy'],
        'context': campaign.record(context / 'campaign.json'), 'parent_campaign': contract['parent_campaign'],
        'admitted_pilot': contract['admitted_pilot'], 'training': campaign.record(training_path),
        'input_audit': training['input_audit'], 'ranking_store': audit['ranking_store'],
        'seed_references': [evidence[row['weight'], row['seed']]['reference'] for row in records],
        'arms': [scalar, candidate], 'scalar_control': scalar, 'frozen_deployed_control': control,
        'selected': selected, 'diagnostics': full_selection.ranking_diagnostic(scalar, candidate),
        'eligible_for_multi_opponent': selected is not None,
        'status': 'full-model-selected-awaiting-strength-evaluation' if selected else 'full-model-offline-rejected',
        'game_banks_opened': False, 'qualification_passed': False, 'campaign_success': False}
    if {key: value for key, value in document.items() if key != 'body_sha256'} != expected:
        raise ValueError('full model selection does not reproduce all six completed training seeds')
    return contract, document


def validate_pilot(root, attempt, full_contract):
    """The admitted pilot and its full phase together count as one attempt."""
    phase = f'attempt-{attempt:03d}-pilot'
    context = root / 'phases' / phase
    contract = attempts.bound(full_contract['pilot_context'], context / 'campaign.json')
    directory = context / phase
    outcome = attempts.bound(full_contract['admitted_pilot'], directory / 'pilot-outcome.json')
    if (outcome.get('admitted') is not True or outcome.get('status') != 'pilot-admitted'
            or outcome.get('campaign_success') is not False):
        raise ValueError('full outcome requires an actual successful prior pilot')
    selection = attempts.bound(outcome['selection'], directory / 'model-selection.json')
    attempts.bound(selection['training'], directory / 'training.json')
    training = attempts.validate_training(context, phase, contract)
    selected = attempts.validate_selection(selection, training, context, phase, contract)
    if selected is None:
        raise ValueError('admitted pilot lacks its reproduced eligible trained seed')
    screen = attempts.validate_screen(directory, outcome, selected, admitted=True)
    return {'attempt': attempt, 'context': context, 'phase': phase, 'contract': contract,
            'outcome': full_contract['admitted_pilot'], 'selection': outcome['selection'],
            'training': selection['training'], 'screen': screen}


def require_absent(directory, *names):
    if any((directory / name).exists() for name in names):
        raise ValueError('later or incomplete stage exists; a completed rejection cannot hide played evidence')


def completed_rejection(root, context, phase, model):
    """Use only the first completed rejecting stage in the actual stage chain."""
    directory = context / phase
    records = {}
    checked_suites = []
    source = None
    # A successful development gate routes to release qualification. This
    # module deliberately has no protected/live result readers or verdicts.
    if model['selected'] is None:
        if model['status'] != 'full-model-offline-rejected' or model['eligible_for_multi_opponent'] is not False:
            raise ValueError('full model has no reproduced offline rejection')
        require_absent(directory, 'search', 'multi-opponent', 'development')
        return 'full-offline', records, source, checked_suites, None
    source = search.validate_selection(root, context, phase)
    records['search'] = campaign.record(directory / 'search/search-selection.json')
    if (source.get('required_ablation_complete') is not True
            or source.get('required_category_profile_complete') is not True
            or source.get('required_profiling_complete') is not True
            or source.get('incomplete_requirements') != []):
        raise ValueError('incomplete profiling or search strength does not close a trained attempt')
    if source['selected'] is None:
        if source['status'] != 'search-strength-rejected' or source['eligible_for_multi_opponent'] is not False:
            raise ValueError('search has no completed source-bound rejection')
        require_absent(directory, 'multi-opponent', 'development')
        return 'search', records, source, checked_suites, None
    for stage in ('screen', 'confirmation'):
        checked = suite._completed_suite(directory / 'multi-opponent' / stage)
        checked_suites.append(checked)
        records[stage] = campaign.record(directory / 'multi-opponent' / stage / 'assessment.json')
        if checked[1]['selection'] != records['search']:
            raise ValueError('completed suite changed the validated selected source')
        if checked[0]['passed'] is False:
            require_absent(directory, 'development')
            if stage == 'screen':
                require_absent(directory, 'multi-opponent/confirmation')
            return stage, records, source, checked_suites, None
        if checked[0]['passed'] is not True:
            raise ValueError('suite does not have a completed boolean verdict')
    result = development.completed_development(context, phase)
    records['development'] = campaign.record(directory / 'development/assessment.json')
    if result.get('passed') is not False or result.get('status') != 'development-rejected':
        raise ValueError('passing or incomplete development is not a completed unsuccessful attempt')
    return 'development', records, source, checked_suites, result


def validated_evidence(root, attempt):
    root = Path(root).resolve()
    if isinstance(attempt, bool) or attempt not in (1, 2, 3):
        raise ValueError('after two unsuccessful trained attempts an intervention binding is required')
    phase = f'attempt-{attempt:03d}-full'
    context = root / 'phases' / phase
    with full_selection.trainer.native_thread_execution_scope():
        contract, model = validate_full_selection(root, context, phase)
        pilot = validate_pilot(root, attempt, contract)
        stage, stages, source, suites, development_result = completed_rejection(root, context, phase, model)
    return {'stage': 'full', 'attempt': attempt, 'context': context, 'phase': phase, 'contract': contract,
            'training': model['training'], 'selection': campaign.record(context / phase / 'full-model-selection.json'),
            'screen': None, 'pilot': pilot, 'rejection_stage': stage, 'stages': stages,
            'source_selection': source, 'suites': suites, 'development': development_result,
            'seed_references': model['seed_references']}


def outcome_body(previous):
    context, phase = previous['context'], previous['phase']
    pilot = previous['pilot']
    return {'schema': campaign.ID + '.full-attempt-outcome.v2',
            'attempt': previous['attempt'], 'context': campaign.record(context / 'campaign.json'),
            'status': 'completed-unsuccessful', 'rejection_stage': previous['rejection_stage'],
            'completed_unsuccessful_trained_attempt': True, 'completed_attempt_count': 1,
            'pilot_and_full_are_one_attempt': True, 'admitted_pilot': pilot['outcome'],
            'pilot_training': pilot['training'], 'pilot_selection': pilot['selection'],
            'training': previous['training'], 'selection': previous['selection'],
            'seed_references': previous['seed_references'], 'stages': previous['stages'],
            'positions': campaign.record(context / phase / 'positions.json'),
            'games': campaign.record(context / phase / 'games.json'),
            'attribution_evidence_scope': 'unprotected-only', 'protected_results_used': False,
            'live_results_used': False, 'campaign_success': False}


def record_outcome(root, attempt):
    previous = validated_evidence(root, attempt)
    path = previous['context'] / previous['phase'] / 'attempt-outcome.json'
    return campaign.seal(path, outcome_body(previous))


def failed_full(root, attempt):
    path = Path(root).resolve() / 'phases' / f'attempt-{attempt:03d}-full' / f'attempt-{attempt:03d}-full/attempt-outcome.json'
    if not path.exists():
        raise ValueError('full attempt is incomplete or has no verified terminal outcome; claims remain spent')
    document = campaign.read(path)
    previous = validated_evidence(root, attempt)
    if {key: value for key, value in document.items() if key != 'body_sha256'} != outcome_body(previous):
        raise ValueError('full terminal outcome differs from completed unprotected evidence')
    previous['outcome'] = campaign.record(path)
    return previous


def collect_fingerprints(previous):
    """Include both corpora and every played variant, candidate and control."""
    values, _ = attempts.collect_fingerprints(previous, expected_games=10000)
    pilot_values, _ = attempts.collect_fingerprints(previous['pilot'])
    for key, fingerprints in pilot_values.items():
        values[key].update(fingerprints)
    if previous['source_selection'] is not None:
        for fps in suite.search_boundaries(previous['source_selection']):
            for domain, value in fps.items():
                values['mixed-development', domain].add(value)
    for _, _, manifest in previous['suites']:
        for binding in manifest['pairs']:
            pair = campaign.read(campaign.verify(binding))
            for arm in ('candidate', 'control'):
                games = suite.checked_games(campaign.verify(pair['arms'][arm]['output']), pair['root'], pair['opponent'])
                for game in games.values():
                    for fps in development.boundaries(game['trajectory'], game['root_transcript']):
                        for domain, value in fps.items():
                            values['mixed-development', domain].add(value)
    if previous['development'] is not None:
        # completed_development independently replays and exactly seals these
        # fingerprint-only records from every native shard and preceding suite.
        for binding in previous['development']['development_exclusions']:
            document = campaign.read(campaign.verify(binding))
            if (document['role'] != 'mixed-development' or document['contains_transcripts'] is not False
                    or document['contains_labels'] is not False or document['contains_metrics'] is not False
                    or document['includes_all_played_postroot_boundaries'] is not True
                    or document['includes_terminal_features'] is not True):
                raise ValueError('development exclusion lost its complete fingerprint-only contract')
            values[document['role'], document['domain']].update(document['fingerprints'])
    return values


def carry_failed_full(root, previous, destination):
    """Create a new-context isolation closure; previous evidence stays sealed."""
    if previous.get('stage') != 'full' or previous['attempt'] not in (1, 2, 3):
        raise ValueError('a verified failed full attempt is required')
    values = collect_fingerprints(previous)
    directory = Path(destination) / 'exclusions' / f'failed-attempt-{previous["attempt"]:03d}'
    sources = {**outcome_body(previous), 'outcome': previous['outcome']}
    artifacts = []
    for ordinal, ((role, domain), members) in enumerate(sorted(values.items())):
        path = directory / f'fingerprints-{ordinal}.json'
        campaign.seal(path, {'schema': campaign.ID + '.failed-full-exclusions.v2', 'role': role, 'domain': domain,
                            'fingerprints': sorted(members), 'sources': sources, 'contains_labels': False,
                            'contains_metrics': False, 'contains_transcripts': False,
                            'source_paths_followed_during_filtering': False})
        artifacts.append(campaign.record(path))
    path = directory / 'index.json'
    campaign.seal(path, {'schema': campaign.ID + '.failed-full-carry.v2', 'attempt': previous['attempt'],
                        'sources': sources, 'artifacts': artifacts, 'completed_attempt_count': 1,
                        'pilot_and_full_are_one_attempt': True, 'pilot_and_full_corpora_included': True,
                        'screen_boundary_coverage': 'all-source-bound-trajectories',
                        'all_executed_unprotected_variants_and_controls_included': True,
                        'new_teacher_labels_required': True, 'early_training_exception_unchanged': True,
                        'validation_never_exempt': True, 'terminal_feature_boundaries_included': True})
    return artifacts, campaign.record(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('command', choices=('record', 'validate'))
    args = parser.parse_args()
    root = args.root.resolve()
    with campaign.lease(root):
        if args.command == 'record':
            result = record_outcome(root, args.attempt)
        else:
            previous = failed_full(root, args.attempt)
            result = campaign.read(campaign.verify(previous['outcome']))
    print(json.dumps({key: result[key] for key in ('attempt', 'status', 'rejection_stage', 'campaign_success')}), flush=True)


if __name__ == '__main__':
    main()
