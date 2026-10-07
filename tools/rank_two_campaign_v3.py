#!/usr/bin/env python3
"""Live-loss research policy. Historical gates and evidence remain immutable."""
from copy import deepcopy
from tools import rank_two_campaign_v2 as prior

SCHEMA = 'papersoccer.rank-two.live-loss.v3'
BASELINE_SHA = prior.BASELINE_SHA
POLICY = {**deepcopy(prior.POLICY), 'schema': SCHEMA + '.policy',
    'candidate_live_admission': 'correctness+safety+targeted+16-root-operational-screen',
    'automatic_baseline_packets': False, 'baseline_research_packet': None,
    'discovery_packet': list('BCCB'), 'replication_packet': list('BCCBBC'),
    'comparison_windows_per_source': 10, 'coverage_windows_per_request': 2,
    'screen_roots': 16, 'screen_games': 64, 'cycle_hypotheses': 2,
    'effort_allocation': {'playing_decisions': .60, 'transfer': .25, 'engineering': .15},
    'minimum_clean_primary_games_per_source': 8, 'review_after_experiments': 2,
    'review_without_live_hours': 48, 'initial_candidate_clocks_ms': [550, 140]}
CORRECTNESS = ('complete_turns', 'both_orientations', 'goals_own_goals', 'blocked_losses',
               'rebounds', 'make_unmake', 'legal_fallback', 'evaluator_decoder_parity',
               'proof_separation', 'sanitizers')
SAFETY = ('isolated', 'cold_start', 'warm_start', 'long_turns', 'full_games', 'hard_limit_margin')


def same_source(value, source):
    return value.get('candidate_source', value.get('source', {})).get('sha256') == source['sha256']


def admission(source, correctness, safety, targeted, screen, roster):
    """Exploration ignores score gates, but requires complete operational evidence."""
    reasons = []
    for label, receipt, checks in [('correctness', correctness, CORRECTNESS), ('safety', safety, SAFETY)]:
        if (receipt.get('passed') is not True or not same_source(receipt, source) or
            any(receipt.get('checks', {}).get(k) is not True for k in checks)):
            reasons.append(label)
    if (targeted.get('complete') is not True or not same_source(targeted, source) or
        targeted.get('legal') is not True or targeted.get('activated') is not True or
        targeted.get('demonstrated_forced_loss') is not False or
        type(targeted.get('own_failures')) is not int or targeted['own_failures'] != 0):
        reasons.append('targeted')
    if (screen.get('complete') is not True or screen.get('stage') != 'screen' or
        not same_source(screen, source) or screen.get('roots') != 16 or
        screen.get('unique_games') != 64 or screen.get('shared_controls') is not True or
        screen.get('fresh') is not True or type(screen.get('failures')) is not int or screen['failures'] != 0 or
        screen.get('roots_per_opponent') != {name: 2 for name in roster}):
        reasons.append('screen')
    return {'passed': not reasons, 'reasons': reasons, 'kind': 'exploratory',
            'promotion_eligible': False, 'strength_qualified': False}


def promotion(source, pilot, development, panel):
    reasons = []
    for name, value, roots in [('pilot', pilot, 64), ('development', development, 256)]:
        a = value.get('assessment', {})
        if (not same_source(value, source) or value.get('stage') != name or value.get('roots') != roots or
            value.get('passed') is not True or a.get('passed') is not True or
            type(a.get('failures')) is not int or a['failures'] != 0): reasons.append(name)
        if name == 'pilot' and a.get('cluster_bootstrap', {}).get('upper_95', -1) < 0: reasons.append('pilot strength')
        if name == 'development' and (a.get('mean_uplift', -1) < .03 or
            len(a.get('opponent_uplift', {})) != 8 or min(a.get('opponent_uplift', {'missing': -1}).values()) < -.05):
            reasons.append('development strength')
    games = panel.get('games', [])
    if (panel.get('complete') is not True or not same_source(panel, source) or
        type(panel.get('own_failures')) is not int or panel['own_failures'] != 0 or len(games) != 12 or
        panel.get('identities_verified') is not True or panel.get('six_opponents_both_colors') is not True):
        reasons.append('panel')
    return {'passed': not reasons, 'reasons': reasons, 'kind': 'promotion',
            'promotion_eligible': not reasons, 'strength_qualified': False}


def decision(kind, cells, candidate_windows, own_failures=0):
    """Frozen cells, equal-cell clean score contrast; no independence/causality claim."""
    if kind not in ('discovery', 'replication'): raise ValueError('comparison kind')
    if own_failures: return {'decision': 'retire', 'reason': 'own-operational-failure', 'sufficient': False}
    sufficient = bool(cells) and all(
        c['B']['games'] >= 8 and c['C']['games'] >= 8 and
        len(set(candidate_windows) & set(c['C']['windows'])) >= min(2, len(candidate_windows)) for c in cells)
    eligible = [c for c in cells if c['B']['games'] and c['C']['games']]
    contrast = (sum(c['C']['score']/c['C']['games'] - c['B']['score']/c['B']['games'] for c in eligible)/len(eligible)
                if eligible else None)
    if kind == 'discovery': choice = 'replicate' if not sufficient or contrast > 0 else 'retire'
    else: choice = 'insufficient coverage' if not sufficient else ('advance to development' if contrast > 0 else 'retire')
    return {'decision': choice, 'sufficient': sufficient, 'contrast': contrast,
            'strength_qualified': False, 'claim': 'Descriptive matched cells; correlated openings and unmatched schedules.'}


def cadence(live=False, long_job=False):
    return 10 if live else 60 if long_job else 20

# Numerical final gates are delegated unchanged; the screen has no strength veto.
def load(path):
    from pathlib import Path
    from tools.rank_two_live_v3 import config
    value = prior.e.read(path)
    return config(Path(path).parent) if value.get('schema') == SCHEMA else prior.load(path)


def assess(results, stage, policy=POLICY):
    value = prior.assess(results, 'pilot' if stage == 'screen' else stage)
    if stage == 'screen': value.update(stage='screen', passed=value['failures'] == 0, strength_eligible=False)
    return value

assess_live_windows = prior.assess_live_windows
