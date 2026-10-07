#!/usr/bin/env python3
"""Live research successor policy; historical local numerical gates stay frozen."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

try:
    from . import rank_two_campaign as prior
except ImportError:
    import rank_two_campaign as prior

e = prior.e
BASELINE_SHA = prior.BASELINE_SHA
SCHEMA = 'papersoccer.rank-two.campaign.v2'
POLICY = {**deepcopy(prior.POLICY),
    'schema': 'papersoccer.rank-two.policy.v2',
    'same_source_ranked_retry': True,
    'distinct_attempt_per_ranked_window': True,
    'baseline_research_packet': ['B', 'B', 'B'],
    'candidate_research_packet': ['B', 'C', 'C', 'B', 'B', 'C'],
    'formal_qualification_windows': 3,
    'formal_qualification_required_rank_two_windows': 2,
    'formal_qualification_blocks_per_source': 1,
    'formal_qualification_predeclared_consecutive': True,
    'candidate_live_admission': 'source-bound safety and pilot',
    'research_rank_is_formal_qualification': False,
}


def initialize(output, prior_root, baseline=None):
    """Copy the frozen roster and bind its historical manifest, without new games."""
    previous_path = Path(prior_root).resolve() / 'campaign.json'
    previous = prior.load(previous_path)
    if baseline is not None and e.raw_record(baseline)['sha256'] != BASELINE_SHA:
        raise ValueError('rank-three rollback source identity changed')
    output = Path(output).resolve()
    if output == previous_path.parent:
        raise ValueError('successor requires a separate output directory')
    roster = {}
    for name, actor in previous['opponents'].items():
        source = output / 'controls' / name / 'submission.cpp'
        e.campaign.immutable(source, e.verify(actor['source']).read_bytes())
        roster[name] = e.descriptor(source, actor['family'], actor['clocks_ms'])
    e.emit(output / 'policy.json', POLICY)
    value = {'schema': SCHEMA, 'policy': e.raw_record(output / 'policy.json'),
             'control': roster['rank_3'], 'rollback': roster['rank_3'], 'opponents': roster,
             'baseline_release': previous['baseline_release'], 'producer': e.raw_record(__file__),
             'predecessor_manifest': e.raw_record(previous_path),
             'confirmation_roots': previous['confirmation_roots'],
             'confirmation_games': previous['confirmation_games']}
    e.emit(output / 'campaign.json', value)
    return value


def load(path):
    value = e.read(path)
    if value.get('schema') != SCHEMA:
        raise ValueError('campaign schema mismatch')
    if e.read(e.verify(value['policy'])) != POLICY:
        raise ValueError('campaign policy changed')
    previous = prior.load(e.verify(value['predecessor_manifest']))
    e.verify(value['producer'])
    if set(value['opponents']) != set(previous['opponents']):
        raise ValueError('rank-two roster changed')
    if value['control'] != value['opponents']['rank_3'] or value['rollback'] != value['control']:
        raise ValueError('rank-two control/rollback changed')
    for name, actor in value['opponents'].items():
        e.descriptor(e.verify(actor['source']), actor['family'], actor['clocks_ms'])
        old = previous['opponents'][name]
        if (actor['source']['sha256'], actor['family'], actor['clocks_ms']) != (old['source']['sha256'], old['family'], old['clocks_ms']):
            raise ValueError('historical actor identity changed')
    for key in ('baseline_release', 'confirmation_roots', 'confirmation_games'):
        if value[key] != previous[key]:
            raise ValueError('historical campaign binding changed: ' + key)
    e.verify(value['baseline_release'])
    return value


def assess(results, stage, policy=POLICY):
    return prior.assess(results, stage, policy)


def assess_live_admission(pilot, safety, candidate_source):
    """Pure candidate admission; callers verify referenced artifacts before loading.

    pilot is a single-candidate strength result, safety a source-bound receipt.
    Passing the pilot permits live research and makes no strength claim.
    """
    assessment = pilot.get('assessment', {})
    if not isinstance(assessment, dict):
        return {'passed': False, 'strength_qualified': False}
    return {'passed': bool(
        pilot.get('stage') == 'pilot' and pilot.get('passed') is True and
        pilot.get('candidate_source') == candidate_source and
        pilot.get('roots') == 8 * POLICY['pilot_roots_per_opponent'] and
        assessment.get('passed') is True and type(assessment.get('failures')) is int and assessment['failures'] == 0 and
        assessment.get('strength_eligible') is False and
        safety.get('passed') is True and safety.get('candidate_source') == candidate_source),
        'strength_qualified': False}


def assess_live_windows(windows, source_sha256, attempt_ids, purpose='qualification',
                        baseline_sha256=BASELINE_SHA):
    """Assess a full predeclared ordered block, never select a favorable subset.

    attempt_ids is the frozen declaration's ordered list, not inferred from results.
    Windows are normalized source-bound outcomes with attempt_id, source_sha256,
    status, percentage, completed_rank, and coverage. Root lifecycle enforcement
    (one block/source, consecutive submissions, declaration before launch) belongs
    to the durable live controller. This pure function cannot prove chronology.
    """
    if purpose not in ('qualification', 'baseline_research', 'candidate_research'):
        raise ValueError('unsupported live purpose')
    for digest in (source_sha256, baseline_sha256):
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('invalid source digest')
    if purpose == 'qualification':
        expected_sources = [source_sha256] * POLICY['formal_qualification_windows']
    else:
        if purpose == 'baseline_research' and source_sha256 != baseline_sha256:
            raise ValueError('baseline packet requires baseline source')
        if purpose == 'candidate_research' and source_sha256 == baseline_sha256:
            raise ValueError('candidate packet requires distinct candidate source')
        expected_sources = [baseline_sha256 if role == 'B' else source_sha256
                            for role in POLICY[purpose + '_packet']]
    if not isinstance(attempt_ids, (list, tuple)) or len(attempt_ids) != len(expected_sources):
        raise ValueError('incorrect predeclared window count')
    if any(not isinstance(a, str) or not a.strip() for a in attempt_ids) or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError('distinct nonempty attempt IDs required')
    if not isinstance(windows, (list, tuple)) or len(windows) != len(expected_sources):
        raise ValueError('full predeclared window count required')
    completed = []
    for window, attempt, source in zip(windows, attempt_ids, expected_sources):
        if not isinstance(window, dict):
            raise ValueError('window must be an outcome object')
        if window.get('attempt_id') != attempt or window.get('source_sha256') != source:
            raise ValueError('window attempt/source sequence changed')
        rank = window.get('completed_rank')
        coverage = window.get('coverage', {})
        if not isinstance(coverage, dict):
            coverage = {}
        completed.append(window.get('status') == 'complete' and
            type(window.get('percentage')) in (int, float) and window['percentage'] == 100 and
            type(rank) is int and rank >= 1 and
            coverage.get('full_window_accounted') is True and
            type(coverage.get('focus_operational_failures')) is int and
            coverage['focus_operational_failures'] == 0)
    rank_two = sum(ok and window['completed_rank'] <= POLICY['target_completed_rank_maximum']
                   for ok, window in zip(completed, windows))
    operationally_complete = all(completed)
    achieved = (purpose == 'qualification' and operationally_complete and
                rank_two >= POLICY['formal_qualification_required_rank_two_windows'])
    return {'purpose': purpose, 'windows': len(windows), 'operationally_complete': operationally_complete,
            'rank_two_windows': rank_two, 'passed': achieved if purpose == 'qualification' else operationally_complete,
            'goal_achieved': False, 'live_qualification_passed': achieved, 'strength_qualified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    value = initialize(args.output, args.prior_root, args.baseline)
    print(json.dumps({'manifest': e.raw_record(args.output / 'campaign.json'),
                      'confirmation_games': value['confirmation_games']}))


if __name__ == '__main__':
    main()
