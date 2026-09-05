#!/usr/bin/env python3
"""Close completed protected/live failures and carry hashes to a fresh attempt.

Qualification verdicts establish completion only. The returned attempt view and
attribution receipt contain unprotected training/search/development evidence;
protected/live scores, game outcomes, and trajectories are never projected into
attribution. Incomplete or precision-inconclusive live evidence cannot close an
attempt. No command executes games, performs network requests, or uploads.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
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
from tools import compact_value_bfm_full_outcome_v2 as full
from tools import compact_value_bfm_live_v2 as live
from tools import compact_value_bfm_protected_v2 as protected
from tools import compact_value_bfm_release_v2 as release

SCHEMA = campaign.ID + '.terminal-attempt-outcome.v2'
DOMAINS = (campaign.legacy.STATE_FINGERPRINT_DOMAIN, campaign.legacy.FEATURE_FINGERPRINT_DOMAIN)


def location(root, attempt):
    if type(attempt) is not int or attempt not in (1, 2, 3):
        raise ValueError('only the three source-bound trained attempt slots are supported')
    root = Path(root).resolve()
    phase = f'attempt-{attempt:03d}-full'
    context = root / 'phases' / phase
    return root, context, phase


def _unprotected_context(root, attempt):
    root, context, phase = location(root, attempt)
    with full.full_selection.trainer.native_thread_execution_scope():
        contract, model = full.validate_full_selection(root, context, phase)
        pilot = full.validate_pilot(root, attempt, contract)
        if model.get('selected') is None or model.get('eligible_for_multi_opponent') is not True:
            raise ValueError('terminal qualification requires an actually eligible full trained model')
        checked_contract, selected, inputs, suites = development.prerequisites(context, phase)
        developed = development.completed_development(context, phase)
    if (checked_contract != contract or developed.get('passed') is not True
            or developed['selected'] != selected
            or inputs['full_model_selection'] != campaign.record(context / phase / 'full-model-selection.json')
            or model['selected']['runtime'] != selected['runtime']):
        raise ValueError('terminal outcome lost its trained full source and passing development chain')
    source = attempts.bound(inputs['selection'], context / phase / 'search/search-selection.json')
    stages = {'search': inputs['selection'], 'screen': inputs['screen'], 'confirmation': inputs['confirmation'],
              'development': campaign.record(context / phase / 'development/assessment.json')}
    previous = {'stage': 'full', 'terminal_outcome': True, 'attempt': attempt, 'context': context,
        'phase': phase, 'contract': contract, 'training': model['training'],
        'selection': campaign.record(context / phase / 'full-model-selection.json'),
        'screen': None, 'pilot': pilot, 'rejection_stage': 'post-development', 'stages': stages,
        'source_selection': source, 'suites': suites, 'development': developed,
        'seed_references': model['seed_references']}
    return previous, selected


def _live_rejection(root, context, phase, selected):
    source_sha = selected['source']['sha256']
    output = live.directory(root, source_sha)
    path = output / 'assessment.json'
    if not path.is_file() or (output / 'assessment-precision-inconclusive.json').exists():
        raise ValueError('live failure needs a completed unambiguous calibrated assessment')
    existing = campaign.read(path)
    actual = live.assess(root, source_sha)
    if (actual != existing or actual.get('schema') != campaign.ID + '.live-assessment.v2'
            or actual.get('status') != 'completed-live-attempt-below-objective'
            or actual.get('campaign_success') is not False or actual.get('exact_games') != 90
            or actual.get('calibration_complete') is not True or type(actual.get('precise_score_required')) is not bool
            or (actual.get('clean_window') is True and actual['precise_score_required'])
            or type(actual.get('clean_window')) is not bool or actual.get('training_eligible') is not False
            or actual.get('identical_source_reupload_allowed') is not False):
        raise ValueError('successful, incomplete, ambiguous or precision-inconclusive live evidence is not a failed attempt')
    auth = live.revalidate_authorization(root, source_sha)
    if (Path(auth['context']) != context or auth['phase'] != phase
            or auth['qualified_source'] != selected['source'] or auth['runtime'] != selected['runtime']
            or auth['payload_sha256'] != selected['payload_sha256']
            or actual['authorization'] != campaign.record(output / 'authorization.json')):
        raise ValueError('terminal live rejection belongs to another full attempt or source')
    return campaign.record(path)


def _validated(root, attempt):
    root, context, phase = location(root, attempt)
    previous, selected = _unprotected_context(root, attempt)
    qualified_path = context / phase / 'protected/assessment.json'
    # A missing final dual assessment cannot be replaced with a partial gate or
    # with an operational exception. The maintained validator reopens all shards.
    existing = campaign.read(qualified_path)
    qualified = protected.validate(root, context, phase)
    if (qualified != existing or qualified['selected'] != selected
            or type(qualified.get('passed')) is not bool or qualified.get('campaign_success') is not False):
        raise ValueError('terminal protected qualification differs from the exact developed source')
    frozen = release.validate(root, context, phase)
    if (qualified['freeze'] != campaign.record(context / phase / 'release/freeze.json')
            or frozen['selected'] != selected or frozen.get('eligible_for_protected') is not True):
        raise ValueError('terminal qualification lost its exact source-specific release freeze')
    live_binding = None
    if qualified['passed']:
        live_binding = _live_rejection(root, context, phase, selected)
    else:
        if qualified.get('status') != 'protected-rejected':
            raise ValueError('protected qualification has no completed rejecting verdict')
        if live.directory(root, selected['source']['sha256']).exists():
            raise ValueError('protected rejection cannot hide later or diagnostic live evidence')
    evidence = {'schema': campaign.ID + '.terminal-completion-evidence.v2',
        'attempt': attempt, 'context': campaign.record(context / 'campaign.json'),
        'release': qualified['freeze'], 'protected': campaign.record(qualified_path), 'live': live_binding,
        'selected_source': selected['source'], 'selected_runtime': selected['runtime'],
        'completed_unsuccessful_trained_attempt': True, 'contains_metrics': False,
        'contains_transcripts': False, 'attribution_may_follow_qualification_references': False}
    return previous, evidence


def outcome_body(previous, evidence):
    # Keep the normal full-attempt topology, including both training phases, so
    # attribution can project the same explicit unprotected fields for all paths.
    body = full.outcome_body(previous)
    for key in ('protected_results_used', 'live_results_used'):
        body.pop(key, None)
    return {**body, 'schema': SCHEMA,
        'terminal_evidence': campaign.record(evidence), 'terminal_verdict_verified': True,
        'qualification_metrics_used_for_attribution': False,
        'qualification_references_followed_for_attribution': False}


def record_outcome(root, attempt):
    previous, evidence = _validated(root, attempt)
    directory = previous['context'] / previous['phase']
    evidence_path = directory / 'terminal-outcome/completion.json'
    campaign.seal(evidence_path, evidence)
    return campaign.seal(directory / 'attempt-outcome.json', outcome_body(previous, evidence_path))


def failed_terminal(root, attempt):
    _, context, phase = location(root, attempt)
    path = context / phase / 'attempt-outcome.json'
    existing = campaign.read(path)
    if existing.get('schema') != SCHEMA:
        raise ValueError('terminal continuation requires its own validated outcome schema')
    previous, evidence = _validated(root, attempt)
    evidence_path = context / phase / 'terminal-outcome/completion.json'
    recorded = attempts.bound(existing['terminal_evidence'], evidence_path)
    if {key: value for key, value in recorded.items() if key != 'body_sha256'} != evidence:
        raise ValueError('terminal completion evidence changed')
    if {key: value for key, value in existing.items() if key != 'body_sha256'} != outcome_body(previous, evidence_path):
        raise ValueError('terminal outcome differs from verified qualification completion')
    # No qualification assessment or live DTO is returned to attribution.
    return {**previous, 'outcome': campaign.record(path)}


def _protected_fingerprints(evidence):
    qualified = campaign.read(campaign.verify(evidence['protected']))
    rows = qualified['protected_exclusions']
    expected = [record for binding in qualified['gates']
                for record in campaign.read(campaign.verify(binding))['protected_exclusions']]
    if rows != expected or len(qualified['gates']) != 2 or len(rows) != 4:
        raise ValueError('terminal carry omitted one protected gate or fingerprint domain')
    values = defaultdict(set)
    for binding in rows:
        item = campaign.read(campaign.verify(binding))
        if (item.get('role') != 'protected' or item.get('domain') not in DOMAINS
                or any(item.get(key) is not True for key in ('includes_all_proposals',
                    'includes_all_played_postroot_boundaries', 'includes_terminal_features'))
                or any(item.get(key) is not False for key in ('contains_transcripts', 'contains_labels', 'contains_metrics'))
                or not isinstance(item.get('fingerprints'), list)
                or any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None
                       for value in item['fingerprints'])):
            raise ValueError('protected carry lost complete fingerprint-only evidence')
        values['protected', item['domain']].update(item['fingerprints'])
    if set(values) != {('protected', domain) for domain in DOMAINS}:
        raise ValueError('protected carry omitted canonical or feature identities')
    return values


def _live_record_fingerprints(record, identity):
    replay = record.get('replay', {}); rules = replay.get('rules_validation', {})
    empty = (replay.get('valid_transcript') == '' and replay.get('valid_turns') == []
        and rules.get('valid_turns') == [] and rules.get('valid_turn_count') == 0
        and rules.get('status') in ('incomplete', 'invalid')
        and record.get('operational', {}).get('classification') == 'operationally-terminated')
    if empty:
        if (record.get('schema') != live.collector.GENERIC_GAME_SCHEMA or record.get('status') != 'accepted'
                or record.get('source_sha256') != identity.source_sha256
                or record.get('focus', {}).get('agent_id') != identity.agent_id
                or record.get('focus', {}).get('submission_id') != identity.submission_id):
            raise ValueError('zero-turn operational live ending lost its source identity')
        fps = campaign.fingerprints(campaign.features.ReplayState())
        return {domain: {value} for domain, value in fps.items()}
    canonical = set(live.collector._canonical_live_boundaries(record, identity=identity))
    fps, _ = attempts.trajectory_fingerprints(replay['valid_transcript'], 0)
    if fps[DOMAINS[0]] != canonical:
        raise ValueError('independent live replay canonical boundaries disagree')
    return fps


def _live_fingerprints(root, evidence):
    if evidence['live'] is None:
        return defaultdict(set)
    selected = evidence['selected_source']; source_sha = selected['sha256']
    auth, submission, reference, receipt = live.live_window(root, source_sha)
    outcome = campaign.read(campaign.verify(evidence['live']))
    if outcome['window'] != campaign.record(reference):
        raise ValueError('terminal live carry lost the completed exact90 window')
    output = live.directory(root, source_sha)
    identity, excluded, _ = live.collector.load_live_identity(
        output / 'collector-submission.json', Path(auth['exclusion_binding']['path']))
    manifest = live.collector.resolve_path(receipt['collector_manifest']['path'])
    checked = live.collector.verify_generic_result({'manifest_path': str(manifest),
        'manifest_sha256': receipt['collector_manifest']['sha256']}, identity=identity,
        registry_sha256=excluded['registry']['sha256'], expected_game_ids=receipt['game_ids'])
    if (len(checked['records']) != 90 or identity.source_sha256 != source_sha
            or any(row['focus']['session_id'] != submission['test_session_handle'] for row in checked['records'])):
        raise ValueError('terminal live carry source, session or exact90 roster changed')
    values = defaultdict(set)
    for record in checked['records']:
        for domain, members in _live_record_fingerprints(record, identity).items():
            values['live', domain].update(members)
    return values


def _release_fingerprints(root, previous):
    """Project the source-bound release helper's played boundaries to hashes."""
    values = defaultdict(set)
    for fps in release.preflight_boundaries(root, previous['context'], previous['phase']):
        for domain, value in fps.items():
            values['mixed-development', domain].add(value)
    return values


def carry_failed_terminal(root, previous, destination):
    validated = failed_terminal(root, previous['attempt'])
    if previous != validated:
        raise ValueError('terminal carry received a modified sanitized attempt view')
    _, evidence = _validated(root, previous['attempt'])
    values = full.collect_fingerprints(validated)
    for additional in (_protected_fingerprints(evidence), _live_fingerprints(root, evidence), _release_fingerprints(root, validated)):
        for key, members in additional.items():
            values[key].update(members)
    directory = Path(destination).resolve() / 'exclusions' / f'failed-attempt-{previous["attempt"]:03d}'
    # References are provenance only. Filtering opens fingerprint arrays, never
    # qualification outcomes, scores, or their recorded trajectories.
    sources = {'outcome': validated['outcome'], 'context': campaign.record(validated['context'] / 'campaign.json'),
               'training': validated['training'], 'selection': validated['selection']}
    artifacts = []
    for ordinal, ((role, domain), members) in enumerate(sorted(values.items())):
        path = directory / f'fingerprints-{ordinal}.json'
        campaign.seal(path, {'schema': campaign.ID + '.terminal-attempt-exclusions.v2',
            'role': role, 'domain': domain, 'fingerprints': sorted(members), 'sources': sources,
            'contains_labels': False, 'contains_metrics': False, 'contains_transcripts': False,
            'source_paths_followed_during_filtering': False})
        artifacts.append(campaign.record(path))
    index = directory / 'index.json'
    campaign.seal(index, {'schema': campaign.ID + '.terminal-attempt-carry.v2',
        'attempt': previous['attempt'], 'sources': sources, 'artifacts': artifacts,
        'completed_attempt_count': 1, 'pilot_and_full_are_one_attempt': True,
        'pilot_and_full_corpora_included': True, 'all_protected_proposals_and_played_boundaries_included': True,
        'completed_live_boundaries_included': evidence['live'] is not None,
        'release_process_boundaries_included': True, 'terminal_feature_boundaries_included': True,
        'new_teacher_labels_required': True, 'early_training_exception_unchanged': True,
        'validation_never_exempt': True, 'protected_never_exempt': True})
    return artifacts, campaign.record(index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--attempt', type=int, required=True, choices=(1, 2, 3))
    parser.add_argument('command', choices=('record', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root.resolve()):
        result = record_outcome(args.root, args.attempt) if args.command == 'record' else campaign.read(
            campaign.verify(failed_terminal(args.root, args.attempt)['outcome']))
    print(json.dumps({key: result[key] for key in ('status', 'completed_attempt_count', 'campaign_success')}))


if __name__ == '__main__':
    main()
