#!/usr/bin/env python3
"""Bind the approved QAT/scale intervention to a fresh third trained attempt.

Preparation requires the frozen, reproduced unprotected attribution after two
completed unsuccessful attempts. Only the existing refined-adaptive-scales-v1
recipe is executable here. Teacher-ranking and search recommendations remain
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
STANDARD = 'standard-v1'


def path(root):
    return Path(root).resolve() / 'interventions/attempt-003.json'


def approved_profile():
    """Use the registered implementation; never construct a new optimizer recipe."""
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


def attribution_scope(document):
    from tools import compact_value_bfm_attribution_v2 as attribution
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


def _body(root, document):
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


def prepare(root):
    """Freeze a concrete approved recipe only after full attribution validation."""
    root = Path(root).resolve()
    if not (root / 'attribution/after-two-attempts.json').is_file():
        raise ValueError('a frozen validated after-two-attempts attribution is required')
    if path(root).exists():
        return validate(root)
    document = validated_attribution(root)
    return campaign.seal(path(root), _body(root, document))


def validate(root):
    """Reproduce the complete authorization boundary without starting any work."""
    root = Path(root).resolve()
    document = campaign.read(path(root))
    expected = _body(root, validated_attribution(root))
    if set(document['producers']) != set(expected['producers']):
        raise ValueError('intervention source closure is incomplete')
    for record in document['producers'].values():
        campaign.verify(record)
    excluded = {'body_sha256', 'producers'}
    if ({key: value for key, value in document.items() if key not in excluded}
            != {key: value for key, value in expected.items() if key not in excluded}):
        raise ValueError('intervention differs from its verified frozen attribution and approved profile')
    return document


def validated_attribution(root):
    from tools import compact_value_bfm_attribution_v2 as attribution
    return attribution.validate(root)


def expected_qat_profile(contract):
    """Check the frozen recipe at downstream entry without rerunning attribution.

The full prepare/validate boundary reproduces the two unsuccessful outcomes.
This reader checks its immutable binding, the exact recipe, and unchanged policy
and inputs. It never opens a protected/live result or selects another recipe.
"""
    attempt = contract.get('attempt')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('command', choices=('prepare', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root):
        result = prepare(args.root) if args.command == 'prepare' else validate(args.root)
    print(json.dumps({'attempt': result['attempt'], 'qat_profile': result['qat_profile'],
                      'new_training_started': False, 'campaign_success': False}), flush=True)


if __name__ == '__main__':
    main()
