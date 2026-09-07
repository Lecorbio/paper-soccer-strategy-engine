#!/usr/bin/env python3
"""Bind the concrete QAT/scale profiles to fresh third or fourth attempts.

Preparation requires the frozen, reproduced unprotected attribution after two
completed unsuccessful attempts. The fourth route additionally requires all
nine completed third-pilot retention failures and its frozen after-three
hypothesis. Teacher-ranking and search recommendations remain
closed until their own approved concrete experiment bindings exist.
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
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_train as trainer

SCHEMA = campaign.ID + '.intervention.v2'
PROFILE = 'refined-adaptive-scales-v1'
RETENTION_PROFILE = 'retention-first-low-rate-v1'
STANDARD = 'standard-v1'


def path(root, attempt=3):
    if type(attempt) is not int or attempt not in (3, 4):
        raise ValueError('only the concrete third and fourth interventions are supported')
    return Path(root).resolve() / f'interventions/attempt-{attempt:03d}.json'


def approved_profile(attempt=3):
    """Use the registered implementation; never construct a new optimizer recipe."""
    path(Path('.'), attempt)
    if attempt == 4:
        if getattr(trainer, 'RETENTION_FIRST_LOW_RATE_QAT_PROFILE', None) != RETENTION_PROFILE:
            raise ValueError('approved retention-first QAT profile is unavailable')
        profile = trainer.qat_profile_contract(RETENTION_PROFILE)
        prior = approved_profile()
        retention = profile['scale_selection'].get('retention_policy', {})
        if (profile['schedule'] != {**prior['schedule'], 'qat_learning_rate': .0000625}
                or profile['quantization'] != prior['quantization']
                or any(profile['scale_selection'].get(key) != value
                       for key, value in prior['scale_selection'].items() if key != 'validation_objective')
                or set(profile['scale_selection']) != set(prior['scale_selection']) | {'retention_policy'}
                or profile['scale_selection']['validation_objective'] !=
                   'retention-feasibility-then-normalized-violation-sum-then-refined-ranking-key-then-lower-scale'
                or not isinstance(retention, dict)
                or retention.get('reference') != 'frozen-pre-QAT-float-validation'
                or retention.get('predicate') != 'existing-offline-advancement-gate'
                or retention.get('failure_score') != 'sum-eight-positive-normalized-violations'
                or retention.get('nonfinite_reference_or_score') != 'reject'):
            raise ValueError('registered retention-first profile changed its approved recipe')
        return profile
    if trainer.REFINED_ADAPTIVE_SCALES_QAT_PROFILE != PROFILE:
        raise ValueError('approved refined QAT profile is unavailable')
    profile = trainer.qat_profile_contract(PROFILE)
    standard = trainer.qat_profile_contract(STANDARD)
    if (profile['schedule'] != standard['schedule'] or profile['quantization'] != standard['quantization']
            or profile['schedule']['float_warmup_epochs'] != 1 or profile['schedule']['qat_epochs'] != 4
            or profile['schedule']['all_layers_trainable_each_qat_epoch'] is not True
            or profile['quantization']['bits'] != 3
            or profile['quantization']['fake_quantized_layers'] != ['w1', 'w2', 'w3']
            or list(trainer.ARCHITECTURES['capacity-12x8'].dimensions) != [6301, 12, 8, 1]):
        raise ValueError('registered intervention changed the approved architecture or training schedule')
    return profile


def attribution_scope(document, attempt=3):
    from tools import compact_value_bfm_attribution_v2 as attribution
    path(Path('.'), attempt)
    if attempt == 4:
        facts = document.get('third_pilot_retention', {})
        if (document.get('schema') != campaign.ID + '.after-three-attribution.v2'
                or document.get('completed_unsuccessful_trained_attempts') != 3
                or [row['attempt'] for row in document['attempts']] != [1, 2, 3]
                or any(row.get('completed_attempt_count') != 1 for row in document['attempts'])
                or document.get('policy') != attribution.AFTER_THREE_POLICY
                or document.get('recommendation') != attribution.AFTER_THREE_RECOMMENDATION
                or any(document.get(key) is not False for key in
                       ('protected_results_used', 'live_results_used', 'new_training_started',
                        'qualification_passed', 'campaign_success'))
                or facts.get('all_nine_retention_failed') is not True
                or facts.get('completed_seed_count') != 9 or facts.get('retention_failed_seed_count') != 9
                or facts.get('attempt') != 3 or facts.get('phase') != 'attempt-003-pilot'
                or document['maintained_profile_menu']['qat_and_scales'].get(RETENTION_PROFILE) != approved_profile(4)):
            raise ValueError('fourth intervention requires the exact after-three retention hypothesis')
        return
    if (document.get('schema') != campaign.ID + '.attribution.v2'
            or document.get('completed_unsuccessful_trained_attempts') != 2
            or [row['attempt'] for row in document['attempts']] != [1, 2]
            or any(row.get('completed_attempt_count') != 1 for row in document['attempts'])
            or document.get('protected_results_used') is not False
            or document.get('live_results_used') is not False
            or document.get('new_training_started') is not False
            or document.get('campaign_success') is not False
            or document['policy'] != attribution.POLICY):
        raise ValueError('intervention requires exactly two verified unprotected failed-attempt attributions')
    recommendation = document['recommendation']
    if (recommendation.get('category') != 'qat-and-scales'
            or recommendation.get('existing_profile_to_consider') != PROFILE
            or recommendation.get('selected_execution_profile') is not None
            or recommendation.get('attempt_three_may_start') is not False):
        raise ValueError('only the approved QAT/scales category has a concrete third-attempt binding')
    if document['maintained_profile_menu']['qat_and_scales'].get(PROFILE) != approved_profile():
        raise ValueError('attribution was frozen against a different QAT profile')


def _body(root, document, *, attempt=3):
    if attempt == 4:
        return _body_four(root, document)
    path(root, attempt)
    from tools import compact_value_bfm_attribution_v2 as attribution
    root = Path(root).resolve()
    attribution_scope(document)
    parent = campaign.read(root / 'campaign.json')
    if parent.get('policy') != campaign.POLICY:
        raise ValueError('intervention cannot change frozen campaign data or qualification budgets')
    initial = parent['inputs']['attempt_one_initial_checkpoint']
    teacher = parent['inputs']['teacher_runtime']
    student = parent['inputs']['attempt_zero_runtime']
    for record in (initial, teacher, student):
        campaign.verify(record)
    previous = []
    for attempt in document['attempts']:
        terminal = attempt['phases'][-1]
        previous.append({'attempt': attempt['attempt'], 'outcome': attempt['outcome'],
                         'training': terminal['training'], 'selection': terminal['selection']})
    return {'schema': SCHEMA, 'attempt': 3, 'category': 'qat-and-scales',
            'parent_campaign': campaign.record(root / 'campaign.json'),
            'attribution': campaign.record(root / 'attribution/after-two-attempts.json'),
            'previous_failed_attempts': previous, 'completed_unsuccessful_trained_attempts': 2,
            'qat_profile': PROFILE, 'qat_profile_contract': approved_profile(),
            'baseline_qat_profile_contract': trainer.qat_profile_contract(STANDARD),
            'initial_float': initial, 'teacher_runtime': teacher, 'pilot_generation_student': student,
            'unchanged_campaign_policy': campaign.POLICY,
            'single_changed_training_setting': 'qat_profile',
            'pilot_search_profile': STANDARD, 'teacher_ranking_profile': STANDARD,
            'all_previous_attempt_exclusions_required': True,
            'fresh_pilot_games': 2000, 'new_accepted_teacher_labels_required': True,
            'pilot_admission_required_before_full': True,
            'protected_metrics_used_for_intervention': False, 'live_metrics_used_for_intervention': False,
            'new_training_started': False, 'qualification_passed': False, 'campaign_success': False,
            'producers': {name: campaign.record(module) for name, module in {
                'intervention': Path(__file__), 'trainer': Path(trainer.__file__),
                'attribution': Path(attribution.__file__), 'campaign': Path(campaign.__file__),
            }.items()}}


def _body_four(root, document):
    from tools import compact_value_bfm_attribution_v2 as attribution
    root = Path(root).resolve(); attribution_scope(document, 4)
    facts = attribution.third_pilot_retention(root, document['third_pilot_retention']['sources'])
    if facts != document['third_pilot_retention']:
        raise ValueError('fourth intervention no longer reproduces all nine third-pilot retention failures')
    third = campaign.read(campaign.verify(facts['sources']['contract']))
    if expected_qat_profile(third) != PROFILE:
        raise ValueError('fourth intervention lost the preceding refined third attempt')
    parent = campaign.read(root / 'campaign.json')
    if parent.get('policy') != campaign.POLICY or third['parent_campaign'] != campaign.record(root / 'campaign.json'):
        raise ValueError('fourth intervention changed the frozen parent campaign')
    inputs = parent['inputs']
    for key in ('attempt_one_initial_checkpoint', 'teacher_runtime', 'attempt_zero_runtime'):
        campaign.verify(inputs[key])
    previous = []
    for row in document['attempts']:
        terminal = row['phases'][-1]
        previous.append({'attempt': row['attempt'], 'outcome': row['outcome'],
                         'training': terminal['training'], 'selection': terminal['selection']})
    if any(previous[-1][key] != facts['sources'][key] for key in ('outcome', 'training', 'selection')):
        raise ValueError('fourth intervention attribution changed its terminal third-pilot bindings')
    return {'schema': SCHEMA, 'attempt': 4, 'category': 'qat-and-scales',
            'parent_campaign': campaign.record(root / 'campaign.json'),
            'attribution': campaign.record(attribution.path(root, after_attempts=3)),
            'previous_failed_attempts': previous, 'completed_unsuccessful_trained_attempts': 3,
            'previous_intervention': third['intervention'],
            'third_pilot_retention_sources': facts['sources'],
            'qat_profile': RETENTION_PROFILE, 'qat_profile_contract': approved_profile(4),
            'baseline_qat_profile_contract': approved_profile(),
            'initial_float': inputs['attempt_one_initial_checkpoint'], 'teacher_runtime': inputs['teacher_runtime'],
            'pilot_generation_student': inputs['attempt_zero_runtime'],
            'unchanged_campaign_policy': campaign.POLICY, 'single_changed_training_setting': 'qat_profile',
            'hypothesis': attribution.AFTER_THREE_RECOMMENDATION['hypothesis'], 'causal_attribution_proven': False,
            'pilot_search_profile': STANDARD, 'teacher_ranking_profile': STANDARD,
            'all_previous_attempt_exclusions_required': True, 'fresh_pilot_games': 2000,
            'new_accepted_teacher_labels_required': True, 'pilot_admission_required_before_full': True,
            'protected_metrics_used_for_intervention': False, 'live_metrics_used_for_intervention': False,
            'new_training_started': False, 'qualification_passed': False, 'campaign_success': False,
            'producers': {name: campaign.record(module) for name, module in {
                'intervention': Path(__file__), 'trainer': Path(trainer.__file__),
                'attribution': Path(attribution.__file__), 'campaign': Path(campaign.__file__),
            }.items()}}


def prepare(root, attempt=3):
    """Freeze a concrete approved recipe only after full attribution validation."""
    root = Path(root).resolve()
    from tools import compact_value_bfm_attribution_v2 as attribution
    output = path(root, attempt)
    if not attribution.path(root, after_attempts=attempt - 1).is_file():
        raise ValueError('a frozen validated attribution is required for this intervention')
    if output.exists():
        return validate(root, attempt)
    document = validated_attribution(root) if attempt == 3 else validated_attribution(root, after_attempts=3)
    return campaign.seal(output, _body(root, document, attempt=attempt))


def validate(root, attempt=3):
    """Reproduce the complete authorization boundary without starting any work."""
    root = Path(root).resolve()
    document = campaign.read(path(root, attempt))
    attribution = validated_attribution(root) if attempt == 3 else validated_attribution(root, after_attempts=3)
    expected = _body(root, attribution, attempt=attempt)
    if set(document['producers']) != set(expected['producers']):
        raise ValueError('intervention source closure is incomplete')
    for record in document['producers'].values():
        campaign.verify(record)
    excluded = {'body_sha256', 'producers'}
    if ({key: value for key, value in document.items() if key not in excluded}
            != {key: value for key, value in expected.items() if key not in excluded}):
        raise ValueError('intervention differs from its verified frozen attribution and approved profile')
    return document


def validated_attribution(root, *, after_attempts=2):
    from tools import compact_value_bfm_attribution_v2 as attribution
    return attribution.validate(root, after_attempts=after_attempts)


def expected_qat_profile(contract):
    """Check the frozen recipe at downstream entry without rerunning attribution.

The full prepare/validate boundary reproduces the two unsuccessful outcomes.
This reader checks its immutable binding, the exact recipe, and unchanged policy
and inputs. It never opens a protected/live result or selects another recipe.
"""
    attempt = contract.get('attempt')
    if type(attempt) is int and attempt == 4:
        return _expected_four(contract)
    if isinstance(attempt, bool) or attempt not in (None, 1, 2, 3):
        raise ValueError('no approved QAT intervention is bound for this attempt')
    profile = contract.get('qat_profile', STANDARD)
    if attempt != 3:
        if (profile != STANDARD or contract.get('intervention') is not None
                or contract.get('qat_profile_contract', trainer.qat_profile_contract(STANDARD))
                != trainer.qat_profile_contract(STANDARD)):
            raise ValueError('the first two standard attempts cannot change the QAT profile')
        return STANDARD
    if (profile != PROFILE or contract.get('qat_profile_contract') != approved_profile()
            or contract.get('policy') != campaign.POLICY
            or contract.get('completed_unsuccessful_trained_attempts') != 2
            or contract.get('phase') not in ('pilot', 'full')):
        raise ValueError('third attempt requires the exact frozen QAT/scales intervention')
    record = contract.get('intervention')
    if not isinstance(record, dict) or not isinstance(record.get('path'), str):
        raise ValueError('third attempt has no immutable intervention binding')
    parent_record = contract['parent_campaign']
    root = Path(parent_record['path']).resolve().parent
    expected_path = path(root)
    if Path(record['path']).absolute() != expected_path or expected_path.resolve() != expected_path:
        raise ValueError('third-attempt intervention is redirected outside its campaign')
    document = campaign.read(campaign.verify(record))
    frozen_attribution = root / 'attribution/after-two-attempts.json'
    if Path(document['attribution']['path']).absolute() != frozen_attribution:
        raise ValueError('intervention attribution belongs to another campaign')
    report = campaign.read(campaign.verify(document['attribution']))
    expected = _body(root, report)
    # Unchanged validation code may run from a newer immutable snapshot. Verify
    # the recorded producer bytes, then compare the substantive frozen contract.
    if set(document['producers']) != set(expected['producers']):
        raise ValueError('intervention source closure is incomplete')
    for item in document['producers'].values():
        campaign.verify(item)
    excluded = {'body_sha256', 'producers'}
    if ({key: value for key, value in document.items() if key not in excluded}
            != {key: value for key, value in expected.items() if key not in excluded}
            or document['parent_campaign'] != parent_record
            or document['initial_float'] != contract['inputs']['attempt_one_initial_checkpoint']
            or document['teacher_runtime'] != contract['inputs']['teacher_runtime']
            or (contract['phase'] == 'pilot'
                and document['pilot_generation_student'] != contract['inputs']['attempt_zero_runtime'])):
        raise ValueError('third-attempt QAT binding changed its frozen inputs or profile')
    closure = contract.get('previous_failed_attempts', [])
    expected_closure = document['previous_failed_attempts']
    if len(closure) != 2 or any(
        {key: row.get(key) for key in ('attempt', 'outcome', 'training', 'selection')} != expected_row
        for row, expected_row in zip(closure, expected_closure)
    ):
        raise ValueError('third-attempt QAT binding dropped a verified prior trained attempt')
    if contract['phase'] == 'pilot' and (
        contract.get('pilot_games') != 2000
        or contract.get('pilot_training_roster') != {'lambdas': [0, .1, .25], 'seeds': list(trainer.FIXED_SEEDS)}
        or contract.get('candidate_lineage') != {
            'mandatory_training': True, 'initial_float': document['initial_float'],
            'generation_student': document['pilot_generation_student'], 'smoke_weights_reused': False}
    ):
        raise ValueError('third pilot changed its fresh data, training roster or original initialization')
    return PROFILE


def _expected_four(contract):
    """Read only the narrow frozen attempt4 binding; never select a new profile."""
    from tools import compact_value_bfm_attribution_v2 as attribution
    if (contract.get('qat_profile') != RETENTION_PROFILE
            or contract.get('qat_profile_contract') != approved_profile(4)
            or contract.get('policy') != campaign.POLICY
            or contract.get('completed_unsuccessful_trained_attempts') != 3
            or contract.get('phase') not in ('pilot', 'full')):
        raise ValueError('no approved fourth attempt without the exact frozen retention-first QAT hypothesis')
    binding = contract.get('intervention')
    if not isinstance(binding, dict) or not isinstance(binding.get('path'), str):
        raise ValueError('fourth attempt has no immutable intervention binding')
    parent = contract['parent_campaign']; root = Path(parent['path']).resolve().parent
    expected_path = path(root, 4)
    if Path(binding['path']).absolute() != expected_path or expected_path.resolve() != expected_path:
        raise ValueError('fourth-attempt intervention left its canonical campaign path')
    document = campaign.read(campaign.verify(binding))
    frozen_attribution = attribution.path(root, after_attempts=3)
    if (Path(document['attribution']['path']).absolute() != frozen_attribution
            or frozen_attribution.resolve() != frozen_attribution):
        raise ValueError('fourth intervention attribution belongs to another campaign')
    report = campaign.read(campaign.verify(document['attribution']))
    expected = _body_four(root, report)
    if set(document['producers']) != set(expected['producers']):
        raise ValueError('fourth intervention source closure is incomplete')
    for item in document['producers'].values():
        campaign.verify(item)
    excluded = {'body_sha256', 'producers'}
    if ({key: value for key, value in document.items() if key not in excluded}
            != {key: value for key, value in expected.items() if key not in excluded}
            or document['parent_campaign'] != parent
            or document['initial_float'] != contract['inputs']['attempt_one_initial_checkpoint']
            or document['teacher_runtime'] != contract['inputs']['teacher_runtime']
            or contract['phase'] == 'pilot' and document['pilot_generation_student'] != contract['inputs']['attempt_zero_runtime']):
        raise ValueError('fourth-attempt QAT binding changed its frozen inputs or profile')
    closure = contract.get('previous_failed_attempts', [])
    if len(closure) != 3 or any(
        {key: row.get(key) for key in ('attempt', 'outcome', 'training', 'selection')} != expected_row
        for row, expected_row in zip(closure, document['previous_failed_attempts'])
    ):
        raise ValueError('fourth-attempt QAT binding dropped a verified prior trained attempt')
    if contract['phase'] == 'pilot' and (
        contract.get('pilot_games') != 2000
        or contract.get('pilot_training_roster') != {'lambdas': [0, .1, .25], 'seeds': list(trainer.FIXED_SEEDS)}
        or contract.get('candidate_lineage') != {
            'mandatory_training': True, 'initial_float': document['initial_float'],
            'generation_student': document['pilot_generation_student'], 'smoke_weights_reused': False}
    ):
        raise ValueError('fourth pilot changed its fresh data, training roster or original initialization')
    return RETENTION_PROFILE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt', type=int, choices=(3, 4), default=3)
    parser.add_argument('command', choices=('prepare', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root):
        result = prepare(args.root, args.attempt) if args.command == 'prepare' else validate(args.root, args.attempt)
    print(json.dumps({'attempt': result['attempt'], 'qat_profile': result['qat_profile'],
                      'new_training_started': False, 'campaign_success': False}), flush=True)


if __name__ == '__main__':
    main()
