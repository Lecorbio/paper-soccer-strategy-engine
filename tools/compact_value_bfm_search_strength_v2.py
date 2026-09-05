#!/usr/bin/env python3
"""Run the frozen full-round search A/B roster on one fresh unprotected bank."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_search_v2 as search
from tools import compact_value_bfm_opponent_suite_v2 as suite
from tools import compact_value_bfm_development_v2 as development

POLICY = {'schema': campaign.ID + '.search-strength-policy.v2',
    'pairs': 500, 'proposals': 4096, 'proposal_attempt_limit': 200000,
    'workers': 1, 'retry_claimed_execution': False,
    'purpose': 'full-round-unprotected-search-ablation',
    'native_configuration': search.maintained.GATE_CONFIGURATION,
    'minimum_candidate_wins': search.maintained.FULL_MINIMUM_WINS,
    'process_timeout_seconds': 70000, 'qualification_passed': False}


def phase_inputs(plan, output):
    context = campaign.verify(plan['context']).parent
    phase = output.parent.parent.name
    return {'context': plan['context'], 'plan': campaign.record(output.parent / 'plan.json'),
        'measurement': campaign.record(output.parent / 'measurement.json'),
        'positions': campaign.record(context / phase / 'positions.json'),
        'games': campaign.record(context / phase / 'games.json')}


def producer_paths():
    return {'driver': Path(__file__), 'campaign': Path(campaign.__file__),
        'suite': Path(suite.__file__), 'openings': Path(campaign.openings.__file__),
        'features': Path(campaign.features.__file__), 'development': Path(development.__file__)}


def isolated_rows(plan, output, candidates):
    inputs = phase_inputs(plan, output)
    context = campaign.verify(inputs['context']).parent
    phase = output.parent.parent.name
    target = defaultdict(set)
    for row in candidates:
        for domain, value in row['fingerprints'].items(): target[domain].add(value)
    collided = suite._current_collisions(context, phase, target)
    return [row for row in candidates if all(value not in collided[domain] for domain, value in row['fingerprints'].items())][:POLICY['pairs']]


def validate_bank(plan, strength_directory):
    output = Path(strength_directory).resolve()
    bank = campaign.read(output / 'bank.json')
    claim = campaign.read(campaign.verify(bank['seed_claim']))
    inputs = phase_inputs(plan, output)
    if (Path(bank['seed_claim']['path']) != output / 'seed-claim.json'
            or claim['inputs'] != inputs or claim['policy'] != POLICY or bank['pairs'] != POLICY['pairs']):
        raise ValueError('search strength seed or phase binding changed')
    for record in claim['inputs'].values(): campaign.verify(record)
    for record in claim['producers'].values(): campaign.verify(record)
    proposals = campaign.read(campaign.verify(bank['proposals']))
    if (Path(bank['proposals']['path']) != output / 'proposals.json'
            or proposals['seed_claim'] != bank['seed_claim'] or len(proposals['rows']) != POLICY['proposals']):
        raise ValueError('search strength proposal provenance changed')
    development.validate_bank_rows(bank['rows'], POLICY['pairs'])
    excluded = campaign.exclusion_sets(campaign.read(campaign.verify(plan['context'])))
    for row in proposals['rows']:
        state = suite.replay(row['transcript'])
        if (campaign.fingerprints(state) != row['fingerprints'] or len(state.used_segments) != row['plies']
                or state.winner is not None or row['opening_id'] != row['fingerprints'][campaign.legacy.STATE_FINGERPRINT_DOMAIN]
                or campaign.rejection(state, 'validation', excluded)):
            raise ValueError('search strength proposal violates frozen isolation')
    if bank['rows'] != isolated_rows(plan, output, proposals['rows']):
        raise ValueError('search strength roots differ from first isolated frozen proposals')
    expected = 'opening_id\ttranscript\n' + ''.join(row['opening_id'] + '\t' + row['transcript'] + '\n' for row in bank['rows'])
    if Path(bank['tsv']['path']) != output / 'bank.tsv' or campaign.verify(bank['tsv']).read_text() != expected:
        raise ValueError('search strength TSV differs from frozen canonical roots')
    search.maintained.gate_support.validate_bank(campaign.verify(bank['tsv']))
    return bank


def prepare_bank(plan, strength_directory):
    output = Path(strength_directory).resolve()
    search.validate_measurement(plan, output.parent)
    if (output / 'bank.json').exists(): return validate_bank(plan, output)
    inputs = phase_inputs(plan, output)
    producers = {name: campaign.copy_checked(path, output / 'producers' / (name + '.py'))
        for name, path in producer_paths().items()}
    claim = campaign.seal(output / 'seed-claim.json', {'schema': campaign.ID + '.search-strength-seed.v2',
        'inputs': inputs, 'policy': POLICY, 'producers': producers})
    claim_record = campaign.record(output / 'seed-claim.json')
    if (output / 'proposals.json').exists():
        proposals = campaign.read(output / 'proposals.json')
        if proposals['seed_claim'] != claim_record: raise ValueError('search proposals changed their seed')
        candidates = proposals['rows']
    else:
        seed = hashlib.sha256(campaign.raw([claim['body_sha256'], 'fresh-search-strength-roots'])).digest()
        excluded = campaign.exclusion_sets(campaign.read(campaign.verify(plan['context'])))
        candidates = []; seen = defaultdict(set)
        for ordinal in range(POLICY['proposal_attempt_limit']):
            generated = campaign.openings.generate_candidate(hashlib.sha256(seed + ordinal.to_bytes(16, 'big')).digest())
            if generated is None: continue
            state, transcript, plies = generated; fps = campaign.fingerprints(state)
            if campaign.rejection(state, 'validation', excluded) or any(value in seen[domain] for domain, value in fps.items()): continue
            for domain, value in fps.items(): seen[domain].add(value)
            candidates.append({'opening_id': fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN],
                'transcript': transcript, 'plies': plies, 'fingerprints': fps})
            if len(candidates) == POLICY['proposals']: break
        else: raise ValueError('search strength proposal budget exhausted')
        campaign.seal(output / 'proposals.json', {'schema': campaign.ID + '.search-strength-proposals.v2',
            'seed_claim': claim_record, 'rows': candidates})
    retained = isolated_rows(plan, output, candidates)
    if len(retained) != POLICY['pairs']: raise ValueError('insufficient isolated search strength roots')
    development.validate_bank_rows(retained)
    tsv = output / 'bank.tsv'
    campaign.once(tsv, ('opening_id\ttranscript\n' + ''.join(row['opening_id'] + '\t' + row['transcript'] + '\n' for row in retained)).encode())
    bank = campaign.seal(output / 'bank.json', {'schema': campaign.ID + '.search-strength-bank.v2',
        'seed_claim': claim_record, 'proposals': campaign.record(output / 'proposals.json'),
        'rows': retained, 'pairs': POLICY['pairs'], 'tsv': campaign.record(tsv)})
    return validate_bank(plan, output)


def compile_command(plan, name, binary):
    return [plan['compiler']['path'], '-std=c++20', '-O3',
        '-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="' + plan['variants'][name]['source']['path'] + '"',
        plan['gate_source']['path'], '-o', str(binary)]


def command(plan, name, binary, bank, raw):
    source = plan['variants'][name]['source']
    return [str(binary), '--bank', bank['tsv']['path'], '--candidate-source', source['path'],
        '--rank4-source', plan['rank4_source']['path'], '--output', str(raw),
        '--expected-bank-sha256', bank['tsv']['sha256'], '--expected-candidate-sha256', source['sha256'],
        '--pair-offset', '0', '--pair-count', str(POLICY['pairs']), '--mode', 'actual-clock',
        '--minimum-candidate-wins', str(POLICY['minimum_candidate_wins']), '--candidate-seed', '1', '--include-trajectories']


def run(root, context, phase):
    plan = search.validate_plan(root, context, phase)
    output = search.directory(context, phase) / 'strength'
    if (output / 'index.json').exists():
        search.validate_strength(plan, output.parent)
        return campaign.read(output / 'index.json')
    if any(list(output.glob(pattern)) for pattern in ('*/claim.json','*/result.json','*/execution.json','*/result.json.trajectories.jsonl')):
        raise ValueError('claimed search strength batch is spent; preserve partial evidence, do not restart')
    if os.getpriority(os.PRIO_PROCESS, 0) != 0: raise ValueError('search strength requires nice zero')
    bank = prepare_bank(plan, output)
    frozen_producers=campaign.read(output/'seed-claim.json')['producers']
    current_producers=producer_paths()
    if set(frozen_producers)!=set(current_producers) or any(
            frozen_producers[name]['sha256']!=campaign.sha(path) for name,path in current_producers.items()):
        raise ValueError('executing search strength producer differs from frozen seed provenance')
    builds = {}
    for name in plan['variants']:
        binary = output / name / 'gate.bin'; binary.parent.mkdir(parents=True, exist_ok=True)
        argv = compile_command(plan, name, binary)
        with (binary.parent / 'build.log').open('wb') as log:
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True, env={**os.environ, **campaign.THREADS})
        builds[name] = (binary, argv)
    executions = {}
    # One worker and the exact same frozen pair order for every source variant.
    for name, (binary, compile_argv) in builds.items():
        destination = binary.parent; raw = destination / 'result.json'; argv = command(plan, name, binary, bank, raw)
        claim = destination / 'claim.json'
        campaign.seal(claim, {'schema': campaign.ID + '.search-strength-claim.v2',
            'plan': campaign.record(output.parent / 'plan.json'), 'variant': name,
            'source': plan['variants'][name]['source'], 'runtime': plan['model']['runtime'],
            'bank': bank['tsv'], 'bank_receipt': campaign.record(output / 'bank.json'),
            'compiler': plan['compiler'], 'workers': 1, 'environment': campaign.THREADS,
            'retry_allowed': False, 'binary': campaign.record(binary), 'gate_source': plan['gate_source'],
            'rank4_source': plan['rank4_source'], 'compile_command': compile_argv, 'command': argv,
            'policy': POLICY, 'process_nice': 0})
        started = time.monotonic()
        with (destination / 'stdout.log').open('xb') as out, (destination / 'stderr.log').open('xb') as err:
            try:
                finished = subprocess.run(argv, stdout=out, stderr=err, env={**os.environ, **campaign.THREADS}, timeout=POLICY['process_timeout_seconds'])
            except subprocess.TimeoutExpired:
                campaign.seal(destination / 'failure.json', {'claim': campaign.record(claim), 'reason': 'process-watchdog',
                    'retry_allowed': False, 'partial_sidecar': str(raw) + '.trajectories.jsonl'})
                raise ValueError('search strength process timed out; claim is spent') from None
        if finished.returncode not in (0, 2) or not raw.exists():
            campaign.seal(destination / 'failure.json', {'claim': campaign.record(claim), 'reason': 'incomplete-native-result',
                'returncode': finished.returncode, 'retry_allowed': False})
            raise ValueError('search strength process ended without complete evidence')
        search.maintained.gate_support.validate_result(raw, expected_bank_sha256=bank['tsv']['sha256'],
            expected_candidate_sha256=plan['variants'][name]['source']['sha256'],
            expected_candidate_search_profile=plan['variants'][name]['metadata']['candidate_search_profile'],
            require_trajectories=True, trajectory_bank=campaign.verify(bank['tsv']))
        execution = destination / 'execution.json'
        campaign.seal(execution, {'schema': campaign.ID + '.search-strength-execution.v2', 'claim': campaign.record(claim),
            'raw': campaign.record(raw), 'returncode': finished.returncode, 'elapsed_seconds': time.monotonic() - started,
            'progress': campaign.record(Path(str(raw) + '.trajectories.jsonl'))})
        executions[name] = campaign.record(execution)
        print(json.dumps({'stage': 'search-strength', 'variant_completed': name}), flush=True)
    result = campaign.seal(output / 'index.json', {'schema': campaign.ID + '.search-strength-index.v2',
        'plan': campaign.record(output.parent / 'plan.json'), 'bank': bank['tsv'],
        'bank_receipt': campaign.record(output / 'bank.json'), 'executions': executions})
    search.validate_strength(plan, output.parent)
    campaign.event(Path(root), 'search-strength-completed', {'index': campaign.record(output / 'index.json'), 'campaign_success': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True); parser.add_argument('command', choices=('prepare', 'run', 'verify'))
    args = parser.parse_args()
    with campaign.lease(args.root.resolve()):
        if args.command == 'run': result = run(args.root.resolve(), args.context.resolve(), args.phase)
        else:
            plan = search.validate_plan(args.root.resolve(), args.context.resolve(), args.phase)
            result = (prepare_bank if args.command == 'prepare' else validate_bank)(plan, search.directory(args.context, args.phase) / 'strength')
    print(json.dumps({'schema': result['schema'], 'qualification_passed': False}))


if __name__ == '__main__': main()
