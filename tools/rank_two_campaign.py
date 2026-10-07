#!/usr/bin/env python3
"""Frozen rank-two policy and manifest; historical campaign producers stay intact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from . import top_three_experiments as e
except ImportError:
    import top_three_experiments as e

SCHEMA = 'papersoccer.rank-two.campaign.v1'
BASELINE_SHA = '78e731a81ce7cc702a200d0832a6c75e967a11b12452051724ab1387c226e4b8'
POLICY = {
    'schema': 'papersoccer.rank-two.policy.v1',
    'target_completed_rank_maximum': 2,
    'source_characters_exclusive_maximum': 100000,
    'hard_response_ms': [1000, 200],
    'initial_candidate_clocks_ms': [550, 140],
    'pilot_roots_per_opponent': 8,
    'development_roots_per_opponent': 32,
    'confirmation_roots_per_opponent': 100,
    'minimum_development_uplift': .03,
    'minimum_opponent_uplift': -.05,
    'confirmation_lower95_exclusive_minimum': 0,
    'maximum_operational_failures': 0,
    'maximum_hypotheses_per_wave': 4,
    'maximum_variants_per_hypothesis': 2,
    'maximum_finalists_per_wave': 2,
    'maximum_diagnostic_positions': 96,
    'maximum_cpu_workers': 10,
    'maximum_memory_bytes': 18 * 1024**3,
    'authoritative_game_workers': 1,
    'same_source_ranked_retry': False,
    'reuse_confirmation_for_tuning': False,
    'paid_compute_authorized': False,
}


def initialize(output, prior_root, baseline):
    output = Path(output).resolve()
    if e.raw_record(baseline)['sha256'] != BASELINE_SHA:
        raise ValueError('rank-three rollback source identity changed')
    historical = e.roster(prior_root)
    actors = {**historical, 'rank_3': e.descriptor(baseline, 'turn_action_v2', [550, 140])}
    roster = {}
    for name, actor in actors.items():
        source = output / 'controls' / name / 'submission.cpp'
        e.campaign.immutable(source, e.verify(actor['source']).read_bytes())
        roster[name] = e.descriptor(source, actor['family'], actor['clocks_ms'])
    e.emit(output / 'policy.json', POLICY)
    value = {'schema': SCHEMA, 'policy': e.raw_record(output / 'policy.json'),
             'control': roster['rank_3'], 'rollback': roster['rank_3'], 'opponents': roster,
             'baseline_release': e.raw_record(baseline), 'producer': e.raw_record(__file__),
             'predecessor_manifest': e.raw_record(Path(prior_root) / 'campaign.json'),
             'confirmation_roots': len(roster) * POLICY['confirmation_roots_per_opponent'],
             'confirmation_games': len(roster) * POLICY['confirmation_roots_per_opponent'] * 4}
    e.emit(output / 'campaign.json', value)
    return value


def load(path):
    value = e.read(path)
    if value['schema'] != SCHEMA:
        raise ValueError('campaign schema mismatch')
    policy = e.read(e.verify(value['policy']))
    if policy != POLICY:
        raise ValueError('campaign policy changed')
    if len(value['opponents']) != 8 or value['opponents'].get('rank_3') != value['control']:
        raise ValueError('rank-two roster/control changed')
    for actor in value['opponents'].values():
        e.descriptor(e.verify(actor['source']), actor['family'], actor['clocks_ms'])
    if value['control']['source']['sha256'] != BASELINE_SHA:
        raise ValueError('comparison baseline changed')
    return value


def assess(results, stage, policy=POLICY):
    """One gate implementation; no inherited development threshold at confirmation."""
    if stage not in ('pilot', 'development', 'confirmation'):
        raise ValueError('unsupported qualification stage')
    result = e.assess(results, 'smoke')
    result.update(stage=stage, passed=False, strength_eligible=stage != 'pilot')
    if result['failures']:
        return result
    floor = min(result['opponent_uplift'].values())
    if stage == 'pilot':
        # Pilot is only a rejection filter, never positive strength evidence.
        result['passed'] = result['cluster_bootstrap']['upper_95'] >= 0
    elif stage == 'development':
        result['passed'] = result['mean_uplift'] >= policy['minimum_development_uplift'] and floor >= policy['minimum_opponent_uplift']
    else:
        result['passed'] = result['cluster_bootstrap']['lower_95'] > policy['confirmation_lower95_exclusive_minimum'] and floor >= policy['minimum_opponent_uplift']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    value = initialize(args.output, args.prior_root, args.baseline)
    print(json.dumps({'manifest': e.raw_record(args.output / 'campaign.json'),
                      'opponents': len(value['opponents']), 'confirmation_games': value['confirmation_games']}))


if __name__ == '__main__':
    main()
