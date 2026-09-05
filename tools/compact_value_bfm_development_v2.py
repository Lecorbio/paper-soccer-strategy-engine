#!/usr/bin/env python3
"""Run a fresh 1,000-game development qualifier after both opponent suites.

A source selected by the search bridge must pass both suites unchanged. A
passing development receipt is never protected or live qualification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_full_selection_v2 as full_selection
from tools import compact_value_bfm_opponent_suite_v2 as suite
from tools import compact_value_bfm_search_v2 as search
from tools import compact_value_bfm_teacher_training as maintained
from submissions.codingame.bots.compact_value_bfm import rank4_gate_support as gate

GATE_SOURCE = campaign.REPO / 'submissions/codingame/bots/compact_value_bfm/rank4_gate_trajectories.cpp'

POLICY = {
    'schema': campaign.ID + '.development-policy.v2', 'pairs': 500,
    'games': 1000, 'minimum_wins': 550, 'minimum_wins_per_color': 265,
    'paired_lower_95_above': .5, 'bootstrap_samples': 20000,
    'bootstrap_unit': 'canonical-root-pair', 'bootstrap_percentile': .025,
    'bootstrap_quantile_method': 'lower', 'failures': 0,
    'candidate_clocks_ms': [800, 155], 'rank4_clocks_ms': [800, 165],
    'external_deadlines_ms': [1000, 200], 'workers': 4, 'pairs_per_shard': 125,
    'proposal_roots': 4096, 'proposal_attempt_limit': 100000,
    'bank_generated_after': 'passing-screen-and-confirmation-of-selected-source',
    'retry_claimed_bank': False,
    'candidate_binding': 'validated-search-selection-and-identical-completed-suites', 'max_turns': 320,
    'shard_watchdog_seconds': 86400,
}


def prerequisites(context, phase):
    """Reconcile both raw suites and bind one actually trained full export."""
    context = Path(context).resolve()
    contract = campaign.read(context / 'campaign.json')
    full_path = context / phase / 'full-model-selection.json'
    selection = campaign.read(full_path)
    if (selection.get('eligible_for_multi_opponent') is not True
            or not selection.get('selected')
            or selection.get('context') != campaign.record(context / 'campaign.json')):
        raise ValueError('development requires a bound eligible full model')
    model = selection['selected']
    full_selection.verify_source_export(model)
    runtime = campaign.read(campaign.verify(model['runtime']))
    if (model['runtime_body_sha256'] != runtime['body_sha256']
            or model['payload_sha256'] != runtime['quantization']['payload_sha256']
            or model.get('canonical_retention_passed') is not True
            or model.get('source_reserve', 0) < 2000):
        raise ValueError('development runtime, payload or retention binding changed')
    training_path = campaign.verify(selection['training'])
    if training_path != context / phase / 'training.json':
        raise ValueError('development training belongs to another phase')
    training = campaign.read(training_path)
    if training.get('smoke') is not False or training.get('mandatory_training_verified') is not True:
        raise ValueError('development requires actual nonsmoke training')
    matches = [row for row in training['results']
               if (row['weight'], row['seed']) == (model['lambda'], model['seed'])]
    if (len(matches) != 1 or any(matches[0][key] != model[key] for key in ('source', 'runtime'))
            or model['seed_reference'] not in selection['seed_references']):
        raise ValueError('development candidate lost its trained seed lineage')
    for item in selection['seed_references']:
        campaign.verify(item)
    inputs = {'context': campaign.record(context / 'campaign.json'),
              'full_model_selection': campaign.record(full_path),
              'positions': campaign.record(context / phase / 'positions.json'),
              'games': campaign.record(context / phase / 'games.json')}
    previous = []
    for stage in ('screen', 'confirmation'):
        directory = context / phase / 'multi-opponent' / stage
        checked = suite._completed_suite(directory)
        if checked[0].get('passed') is not True:
            raise ValueError('development requires passing suites of the exact selected source')
        if previous and checked[1]['selection'] != previous[0][1]['selection']:
            raise ValueError('development suite source selections differ')
        previous.append(checked)
        inputs[stage] = campaign.record(directory / 'assessment.json')
    inputs['selection'] = previous[0][1]['selection']
    source_selection = campaign.read(campaign.verify(inputs['selection']))
    parent_root = campaign.verify(contract['parent_campaign']).parent
    if source_selection != search.validate_selection(parent_root, context, phase):
        raise ValueError('development suite selection differs from validated search selection')
    if (source_selection.get('full_model_selection') != inputs['full_model_selection']
            or not source_selection.get('selected')):
        raise ValueError('development requires the validated search source selection and trained model')
    source = source_selection['selected']
    profile = source_selection.get('candidate_search_profile', source.get('candidate_search_profile'))
    if source.get('runtime') != model['runtime'] or profile not in gate.SEARCH_PROFILES:
        raise ValueError('development search selection changed trained runtime or profile')
    candidate = campaign.verify(source['source'])
    source_bytes = candidate.read_bytes(); source_bytes.decode('ascii')
    if 95000 - len(source_bytes) < 2000:
        raise ValueError('development selected source lost required reserve')
    for _, _, manifest in previous:
        claim = campaign.read(campaign.verify(manifest['claim']))
        if claim['sources']['candidate'] != source['source']:
            raise ValueError('development suites executed a different selected source')
    selected = {**source, 'source_reserve': 95000 - len(source_bytes),
                'candidate_search_profile': profile, 'source_selection': inputs['selection']}
    return contract, selected, inputs, previous


def boundaries(transcript, root_transcript):
    """Yield root, each accepted postroot turn, and terminal feature identity."""
    actions, prefix = transcript.split('/'), root_transcript.split('/')
    if actions[:len(prefix)] != prefix:
        raise ValueError('played trajectory lost its frozen prefix')
    state = campaign.features.ReplayState()
    for turn, action in enumerate(actions):
        if turn >= len(prefix):
            yield campaign.fingerprints(state)
        campaign.features.apply_complete_turn(state, state.to_move, action)
    yield campaign.fingerprints(state)


def suite_boundaries(previous):
    # The suite selection binds all earlier search variants, including variants
    # rejected by search selection; those played states still cannot be reused.
    source_selection = campaign.read(campaign.verify(previous[0][1]['selection']))
    strength = campaign.read(campaign.verify(source_selection['strength']))
    for binding in strength['executions'].values():
        execution = campaign.read(campaign.verify(binding))
        document = json.loads(campaign.verify(execution['raw']).read_bytes())
        for game in document['games']:
            yield from boundaries(game['transcript'], game['root_transcript'])
    for _, _, manifest in previous:
        for binding in manifest['pairs']:
            pair = campaign.read(campaign.verify(binding))
            for arm in ('candidate', 'control'):
                for game in suite.evaluation.load_games(campaign.verify(pair['arms'][arm]['output'])).values():
                    yield from boundaries(game['trajectory'], game['root_transcript'])


def collisions(context, phase, target, previous):
    result = suite._current_collisions(context, phase, target)
    for fps in suite_boundaries(previous):
        for domain, value in fps.items():
            if value in target.get(domain, set()):
                result[domain].add(value)
    return result


def validate_bank_rows(rows, pairs=500):
    if len(rows) != pairs:
        raise ValueError('development requires exactly 500 canonical pairs')
    seen = defaultdict(set)
    for row in rows:
        state = suite.replay(row['transcript'])
        fps = campaign.fingerprints(state)
        if (state.winner is not None or len(state.used_segments) < 12
                or row['plies'] != len(state.used_segments) or row['fingerprints'] != fps
                or row['opening_id'] != fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN]):
            raise ValueError('development bank state, progress or canonical ID changed')
        for domain, value in fps.items():
            if value in seen[domain]:
                raise ValueError('development reused a canonical state or feature root')
            seen[domain].add(value)
    return dict(seen)


def validate_bank(context, phase, bank, inputs, previous):
    directory = Path(context) / phase / 'development'
    claim_path = directory / 'seed-claim.json'
    if bank['claim'] != campaign.record(claim_path) or bank.get('pairs') != POLICY['pairs']:
        raise ValueError('development bank seed belongs to another execution')
    claim = campaign.read(claim_path)
    if claim['inputs'] != inputs or claim['policy'] != POLICY:
        raise ValueError('development seed claim inputs or policy changed')
    for item in claim['inputs'].values():
        campaign.verify(item)
    for item in claim['producers'].values():
        campaign.verify(item)
    target = validate_bank_rows(bank['rows'])
    expected_tsv = 'opening_id\ttranscript\n' + ''.join(
        row['opening_id'] + '\t' + row['transcript'] + '\n' for row in bank['rows'])
    if (Path(bank['tsv']['path']) != directory / 'bank.tsv'
            or campaign.verify(bank['tsv']).read_text() != expected_tsv):
        raise ValueError('development TSV differs from frozen canonical rows')
    gate.validate_bank(campaign.verify(bank['tsv']))
    proposals = campaign.read(campaign.verify(bank['proposals']))
    if (Path(bank['proposals']['path']) != directory / 'proposals.json'
            or proposals['claim'] != bank['claim'] or len(proposals['rows']) != POLICY['proposal_roots']):
        raise ValueError('development proposal provenance changed')
    if any(row not in proposals['rows'] for row in bank['rows']):
        raise ValueError('development bank introduced an unfrozen proposal')
    excluded = campaign.exclusion_sets(campaign.read(Path(context) / 'campaign.json'))
    if any(campaign.rejection(suite.replay(row['transcript']), 'validation', excluded) for row in bank['rows']):
        raise ValueError('development root overlaps a prior excluded state')
    if any(collisions(context, phase, target, previous).values()):
        raise ValueError('development root overlaps current data or played suite states')
    return bank


def prepare_bank(context, phase, ready=None):
    context = Path(context).resolve()
    contract, selected, inputs, previous = ready or prerequisites(context, phase)
    directory = context / phase / 'development'
    reference = directory / 'bank.json'
    if reference.exists():
        return validate_bank(context, phase, campaign.read(reference), inputs, previous)
    producers = {name: campaign.copy_checked(path, directory / 'provenance' / (name + '.py'))
                 for name, path in producer_paths().items()}
    claim = campaign.seal(directory / 'seed-claim.json', {
        'schema': campaign.ID + '.development-seed-claim.v2', 'inputs': inputs,
        'policy': POLICY, 'producers': producers})
    proposal_path = directory / 'proposals.json'
    if proposal_path.exists():
        proposals = campaign.read(proposal_path)
        if proposals['claim'] != campaign.record(directory / 'seed-claim.json'):
            raise ValueError('development proposal seed changed')
        candidates = proposals['rows']
    else:
        excluded = campaign.exclusion_sets(contract)
        seed = hashlib.sha256(campaign.raw([claim['body_sha256'], 'fresh-development-roots'])).digest()
        candidates, seen = [], defaultdict(set)
        for attempt in range(POLICY['proposal_attempt_limit']):
            generated = campaign.openings.generate_candidate(hashlib.sha256(seed + attempt.to_bytes(16, 'big')).digest())
            if generated is None:
                continue
            state, transcript, plies = generated
            fps = campaign.fingerprints(state)
            if (campaign.rejection(state, 'validation', excluded)
                    or any(value in seen[domain] for domain, value in fps.items())):
                continue
            for domain, value in fps.items():
                seen[domain].add(value)
            candidates.append({'opening_id': fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN],
                               'transcript': transcript, 'plies': plies, 'fingerprints': fps})
            if len(candidates) == POLICY['proposal_roots']:
                break
        else:
            raise ValueError('development frozen root proposal budget exhausted')
        campaign.seal(proposal_path, {'schema': campaign.ID + '.development-proposals.v2',
            'claim': campaign.record(directory / 'seed-claim.json'), 'rows': candidates})
    target = {domain: {row['fingerprints'][domain] for row in candidates} for domain in candidates[0]['fingerprints']}
    collided = collisions(context, phase, target, previous)
    retained = [row for row in candidates if all(value not in collided[domain]
                for domain, value in row['fingerprints'].items())][:POLICY['pairs']]
    validate_bank_rows(retained)
    tsv = directory / 'bank.tsv'
    campaign.once(tsv, ('opening_id\ttranscript\n' + ''.join(
        row['opening_id'] + '\t' + row['transcript'] + '\n' for row in retained)).encode('ascii'))
    gate.validate_bank(tsv)
    return campaign.seal(reference, {'schema': campaign.ID + '.development-bank.v2',
        'claim': campaign.record(directory / 'seed-claim.json'), 'tsv': campaign.record(tsv),
        'proposals': campaign.record(proposal_path), 'rows': retained, 'pairs': POLICY['pairs']})


def compile_command(directory, compiler):
    return [str(compiler), '-std=c++20', '-O3',
            '-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="' + str(directory / 'candidate.cpp') + '"',
            str(directory / 'compact_value_bfm/rank4_gate_trajectories.cpp'), '-o', str(directory / 'gate.bin')]


def producer_paths():
    return {'driver': Path(__file__), **{name: Path(module.__file__) for name, module in (
        ('campaign', campaign), ('suite', suite), ('openings', campaign.openings),
        ('features', campaign.features), ('bootstrap', maintained), ('validator', gate),
        ('search', search), ('full_selection', full_selection))}}


def validate_build(binding, contract, selected):
    path = campaign.verify(binding)
    directory = path.parent
    build = campaign.read(path)
    expected_paths = {'candidate': directory / 'candidate.cpp',
                      'gate': directory / 'compact_value_bfm/rank4_gate_trajectories.cpp',
                      'rank4': directory / 'rank_4/submission.cpp'}
    if (path.name != 'build.json' or build['candidate'] != selected['source']
            or build['runtime'] != selected['runtime'] or build['compiler'] != contract['compiler']
            or build['policy'] != POLICY or Path(build['binary']['path']) != directory / 'gate.bin'
            or build['gate_original'] != campaign.record(GATE_SOURCE)
            or set(build['sources']) != set(expected_paths)):
        raise ValueError('development compiler, model, binary or source closure changed')
    for name, expected in expected_paths.items():
        if campaign.verify(build['sources'][name]) != expected:
            raise ValueError('development compile source path differs')
    for name, original in (('candidate', selected['source']),
                           ('rank4', contract['opponents']['rank_4']['submission.cpp']),
                           ('gate', build['gate_original'])):
        campaign.verify(original)
        if any(build['sources'][name][field] != original[field] for field in ('sha256', 'bytes')):
            raise ValueError('development compiled source copy differs')
    if build['sources']['rank4']['sha256'] != gate.RANK4_SHA256:
        raise ValueError('development historical Rank4 changed')
    compiler = campaign.verify(build['compiler'])
    campaign.verify(build['binary']); campaign.verify(build['log'])
    if build['command'] != compile_command(directory, compiler):
        raise ValueError('development compile command changed')
    return build


def build_gate(directory, contract, selected):
    output = directory / 'build'
    path = output / 'build.json'
    if path.exists():
        validate_build(campaign.record(path), contract, selected)
        return campaign.record(path)
    source = GATE_SOURCE
    sources = {
        'candidate': campaign.copy_checked(campaign.verify(selected['source']), output / 'candidate.cpp'),
        'gate': campaign.copy_checked(source, output / 'compact_value_bfm/rank4_gate_trajectories.cpp'),
        'rank4': campaign.copy_checked(campaign.verify(contract['opponents']['rank_4']['submission.cpp']), output / 'rank_4/submission.cpp')}
    compiler = campaign.verify(contract['compiler'])
    command = compile_command(output, compiler)
    with (output / 'build.log').open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env={**os.environ, **campaign.THREADS})
    campaign.seal(path, {'schema': campaign.ID + '.development-build.v2',
        'candidate': selected['source'], 'runtime': selected['runtime'], 'sources': sources,
        'gate_original': campaign.record(source), 'compiler': contract['compiler'],
        'command': command, 'binary': campaign.record(output / 'gate.bin'),
        'log': campaign.record(output / 'build.log'), 'policy': POLICY})
    validate_build(campaign.record(path), contract, selected)
    return campaign.record(path)


def shard_command(directory, ordinal, bank, build):
    return [build['binary']['path'], '--bank', bank['tsv']['path'],
        '--candidate-source', build['sources']['candidate']['path'],
        '--rank4-source', build['sources']['rank4']['path'],
        '--output', str(directory / 'shards' / str(ordinal) / 'result.json'),
        '--expected-bank-sha256', bank['tsv']['sha256'],
        '--expected-candidate-sha256', build['candidate']['sha256'],
        '--pair-offset', str(ordinal * POLICY['pairs_per_shard']),
        '--pair-count', str(POLICY['pairs_per_shard']), '--mode', 'actual-clock',
        '--max-turns', '320', '--include-trajectories']


def execute_shard(directory, ordinal, command):
    output = directory / 'shards' / str(ordinal)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (output / 'stdout.log').open('xb') as out, (output / 'stderr.log').open('xb') as err:
        try:
            finished = subprocess.run(command, stdout=out, stderr=err, env={**os.environ, **campaign.THREADS},
                                      timeout=POLICY['shard_watchdog_seconds'])
            returncode, timeout = finished.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timeout = None, True
    result = output / 'result.json'
    progress = output / 'result.json.trajectories.jsonl'
    campaign.seal(output / 'execution.json', {'schema': campaign.ID + '.development-shard.v2',
        'ordinal': ordinal, 'command': command, 'returncode': returncode, 'timeout': timeout,
        'elapsed_seconds': time.monotonic() - started,
        'raw': campaign.record(result) if result.exists() else None,
        'trajectory_progress': campaign.record(progress) if progress.exists() else None,
        'stdout': campaign.record(output / 'stdout.log'), 'stderr': campaign.record(output / 'stderr.log')})
    return campaign.record(output / 'execution.json')


def assess_games(games, rows, seed_material):
    expected = [(index, row['opening_id'], color) for index, row in enumerate(rows) for color in (0, 1)]
    if len(rows) != 500 or [(row['pair_index'], row['opening_id'], row['candidate_player']) for row in games] != expected:
        raise ValueError('development game schedule differs from 500 paired canonical roots')
    wins = [sum(game['failure'] is None and game['winner'] == color
                for game in games if game['candidate_player'] == color) for color in (0, 1)]
    failures = sum(game['failure'] is not None for game in games)
    lower = maintained.paired_bootstrap_lower_95({'games': games}, seed_material=seed_material, samples=20000)
    return {'games': len(games), 'pairs': len(rows), 'candidate_wins': sum(wins),
            'candidate_wins_by_color': wins, 'failures': failures, 'paired_lower_95': lower,
            'passed': sum(wins) >= 550 and min(wins) >= 265 and lower > .5 and failures == 0}


def finalize(context, phase):
    """Validate finished shard receipts; never execute or retry a game."""
    context = Path(context).resolve()
    ready = prerequisites(context, phase)
    contract, selected, inputs, previous = ready
    directory = context / phase / 'development'
    bank = validate_bank(context, phase, campaign.read(directory / 'bank.json'), inputs, previous)
    claim_path = directory / 'execution-claim.json'
    claim = campaign.read(claim_path)
    if (claim['bank'] != campaign.record(directory / 'bank.json') or claim['inputs'] != inputs
            or claim['selected'] != selected or claim['policy'] != POLICY
            or claim['producers'] != campaign.read(directory / 'seed-claim.json')['producers']
            or Path(claim['build']['path']) != directory / 'build/build.json'
            or claim.get('nice') != 0 or claim.get('threads') != campaign.THREADS):
        raise ValueError('development execution claim changed its source, bank or execution policy')
    for item in claim['producers'].values():
        campaign.verify(item)
    build = validate_build(claim['build'], contract, selected)
    commands = [shard_command(directory, ordinal, bank, build) for ordinal in range(POLICY['workers'])]
    if claim['commands'] != commands:
        raise ValueError('development frozen execution commands changed')
    receipts, games, shared_configuration = [], [], None
    for ordinal, command in enumerate(commands):
        output = directory / 'shards' / str(ordinal)
        path = output / 'execution.json'
        receipt = campaign.read(path)
        if (receipt['ordinal'] != ordinal or receipt['command'] != command or receipt['timeout']
                or receipt['returncode'] not in (0, 2) or not receipt['raw']
                or Path(receipt['raw']['path']) != output / 'result.json'):
            raise ValueError('development shard is incomplete or belongs to another execution; claim is spent')
        for key in ('stdout', 'stderr', 'trajectory_progress'):
            if receipt.get(key):
                campaign.verify(receipt[key])
        checked = gate.validate_result(campaign.verify(receipt['raw']),
            expected_bank_sha256=bank['tsv']['sha256'], expected_candidate_sha256=selected['source']['sha256'],
            expected_candidate_search_profile=selected['candidate_search_profile'], require_trajectories=True,
            trajectory_bank=campaign.verify(bank['tsv']))
        config, bindings = checked['config'], checked['bindings']
        if (config['mode'] != 'actual-clock' or config['pair_offset'] != ordinal * 125 or config['pair_count'] != 125
                or config['max_turns'] != 320 or config['minimum_candidate_wins'] != -1 or config['minimum_wins_per_color'] != -1
                or bindings['candidate_runtime_body_sha256'] != selected['runtime_body_sha256']
                or bindings['candidate_payload_sha256'] != selected['payload_sha256']
                or receipt['returncode'] != (0 if checked['result']['passed'] else 2)):
            raise ValueError('development raw configuration, model payload or return code changed')
        if (bindings['candidate_source_bytes'] != build['sources']['candidate']['bytes']
                or bindings['rank4_source_bytes'] != build['sources']['rank4']['bytes']):
            raise ValueError('development raw source sizes differ from compiled inputs')
        configuration = {key: value for key, value in config.items() if key != 'pair_offset'}
        if shared_configuration is not None and configuration != shared_configuration:
            raise ValueError('development shards used different native search configurations')
        shared_configuration = configuration
        receipts.append(campaign.record(path)); games.extend(checked['games'])
    summary = assess_games(games, bank['rows'], claim['body_sha256'])
    execution_path = directory / 'execution.json'
    campaign.seal(execution_path, {'schema': campaign.ID + '.development-execution.v2',
        'claim': campaign.record(claim_path), 'shards': receipts, 'result': summary})
    values = defaultdict(set)
    for fps in suite_boundaries(previous):
        for domain, value in fps.items(): values[domain].add(value)
    for game in games:
        for fps in boundaries(game['transcript'], game['root_transcript']):
            for domain, value in fps.items(): values[domain].add(value)
    exclusions = []
    for ordinal, (domain, fingerprints) in enumerate(sorted(values.items())):
        path = directory / ('played-exclusion-' + str(ordinal) + '.json')
        campaign.seal(path, {'schema': campaign.ID + '.development-played-exclusions.v2',
            'role': 'mixed-development', 'domain': domain, 'fingerprints': sorted(fingerprints),
            'execution': campaign.record(execution_path), 'prior_suites': {key: inputs[key] for key in ('screen', 'confirmation')},
            'contains_transcripts': False, 'contains_labels': False, 'contains_metrics': False,
            'includes_all_played_postroot_boundaries': True, 'includes_terminal_features': True})
        exclusions.append(campaign.record(path))
    return campaign.seal(directory / 'assessment.json', {'schema': campaign.ID + '.development-assessment.v2',
        'execution': campaign.record(execution_path), 'selected': selected, **summary,
        'development_exclusions': exclusions, 'policy': POLICY,
        'status': 'development-passed-awaiting-exact-source-ci-and-protected-gates' if summary['passed'] else 'development-rejected',
        'protected_gates_passed': False, 'campaign_success': False})


def completed_development(context, phase):
    path = Path(context).resolve() / phase / 'development/assessment.json'
    expected = campaign.read(path)
    reproduced = finalize(context, phase)
    if expected != reproduced:
        raise ValueError('development assessment does not reproduce its raw source-bound evidence')
    return reproduced


def run(root, context, phase):
    root, context = Path(root).resolve(), Path(context).resolve()
    directory = context / phase / 'development'
    if (directory / 'assessment.json').exists():
        return completed_development(context, phase)
    if (directory / 'execution-claim.json').exists() or any((directory / 'shards').glob('*/result.json')):
        raise ValueError('claimed development bank is spent; assess finished receipts without rerunning games')
    ready = prerequisites(context, phase)
    contract, selected, inputs, _ = ready
    if campaign.verify(contract['parent_campaign']).parent != root:
        raise ValueError('development campaign parent changed')
    bank = prepare_bank(context, phase, ready)
    build_binding = build_gate(directory, contract, selected)
    build = validate_build(build_binding, contract, selected)
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError('actual-clock development requires nice zero')
    commands = [shard_command(directory, ordinal, bank, build) for ordinal in range(POLICY['workers'])]
    producers = campaign.read(directory / 'seed-claim.json')['producers']
    current_producers = producer_paths()
    if (set(producers) != set(current_producers) or any(
            any(producers[name][field] != campaign.record(path)[field] for field in ('bytes', 'sha256'))
            for name, path in current_producers.items())):
        raise ValueError('development execution producers changed after bank preparation; use the frozen source snapshot')
    campaign.seal(directory / 'execution-claim.json', {'schema': campaign.ID + '.development-execution-claim.v2',
        'inputs': inputs, 'selected': selected, 'bank': campaign.record(directory / 'bank.json'),
        'build': build_binding, 'commands': commands, 'policy': POLICY, 'producers': producers,
        'nice': 0, 'threads': campaign.THREADS})
    with concurrent.futures.ThreadPoolExecutor(max_workers=POLICY['workers']) as pool:
        futures = [pool.submit(execute_shard, directory, ordinal, command) for ordinal, command in enumerate(commands)]
        for future in futures:
            future.result()
    result = finalize(context, phase)
    campaign.event(root, 'development-qualifier-completed', {'assessment': campaign.record(directory / 'assessment.json'),
        'passed': result['passed'], 'campaign_success': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('command', choices=('prepare', 'run', 'assess'))
    args = parser.parse_args()
    contract = campaign.read(args.context.resolve() / 'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent != args.root.resolve():
        raise ValueError('development campaign parent changed')
    with campaign.lease(args.root.resolve()):
        if args.command == 'prepare':
            result = prepare_bank(args.context, args.phase)
            print(json.dumps({'pairs': result['pairs'], 'games_started': False}), flush=True)
            return
        result = run(args.root, args.context, args.phase) if args.command == 'run' else finalize(args.context, args.phase)
    print(json.dumps({key: result[key] for key in ('status', 'passed', 'candidate_wins', 'paired_lower_95', 'campaign_success')}), flush=True)


if __name__ == '__main__':
    main()
