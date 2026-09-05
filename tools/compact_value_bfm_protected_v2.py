#!/usr/bin/env python3
"""Source-frozen, sequential protected Rank4 gates for the trained-v2 campaign.

Only ``run`` materializes roots or starts games. Every started five-pair shard
is spent unless its complete immutable raw evidence validates. Completed shards
can be reused; missing or invalid receipts never authorize another execution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
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
from tools import compact_value_bfm_development_v2 as development
from tools import compact_value_bfm_stream_v2 as stream
from submissions.codingame.bots.compact_value_bfm import rank4_gate_support as gate

POLICY = {
    'schema': campaign.ID + '.protected-policy.v2', 'gates': ['a', 'b'],
    'pairs_per_gate': 500, 'games_per_gate': 1000, 'shards': 100,
    'pairs_per_shard': 5, 'workers': 4, 'minimum_wins': 527,
    'minimum_wins_per_color': 260, 'failures': 0,
    'candidate_clocks_ms': [800, 155], 'rank4_clocks_ms': [800, 165],
    'external_deadlines_ms': [1000, 200], 'max_turns': 320,
    'seed_bytes': 32, 'seed_source': 'os.urandom', 'proposal_roots': 4096,
    'proposal_attempt_limit': 100000, 'shard_watchdog_seconds': 86400,
    'missing_or_invalid_started_shard': 'spent-no-retry',
    'gate_b_after': 'all-gate-a-shards-validated-regardless-of-score',
    'gate_b_excludes': 'all-gate-a-proposals-and-played-postroot-boundaries',
    'source_changes_after_freeze': False,
}
DOMAINS = (campaign.legacy.STATE_FINGERPRINT_DOMAIN, campaign.legacy.FEATURE_FINGERPRINT_DOMAIN)
GATE_SOURCE = development.GATE_SOURCE


class SpentShardError(ValueError):
    """A claimed shard may never be run again without a valid completion."""



def _seal_or_validate(path, expected, *, create):
    if create:
        return campaign.seal(path, expected)
    actual = campaign.read(path)
    if {key: value for key, value in actual.items() if key != 'body_sha256'} != expected:
        raise ValueError('protected immutable evidence does not reproduce')
    return actual


def prerequisites(root, context, phase):
    # Lazy import keeps this bridge's pure validators usable independently.
    from tools import compact_value_bfm_release_v2 as release
    root, context = Path(root).resolve(), Path(context).resolve()
    frozen = release.validate(root, context, phase)
    freeze_path = context / phase / 'release/freeze.json'
    if frozen != campaign.read(freeze_path) or frozen.get('eligible_for_protected') is not True:
        raise ValueError('protected source freeze did not reproduce release validation')
    contract = campaign.read(context / 'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent != root:
        raise ValueError('protected campaign parent changed')
    selected = frozen['selected']
    source = campaign.verify(selected['source'])
    source.read_bytes().decode('ascii')
    runtime = campaign.read(campaign.verify(selected['runtime']))
    if (95000 - source.stat().st_size < 2000
            or selected['runtime_body_sha256'] != runtime['body_sha256']
            or selected['payload_sha256'] != runtime['quantization']['payload_sha256']
            or selected['candidate_search_profile'] not in gate.SEARCH_PROFILES):
        raise ValueError('protected source, runtime, reserve or search profile changed')
    assessment = development.completed_development(context, phase)
    if (assessment.get('passed') is not True
            or any(assessment['selected'][key] != selected[key] for key in
                   ('source', 'runtime', 'runtime_body_sha256', 'payload_sha256', 'candidate_search_profile'))
            or frozen['development'] != campaign.record(context / phase / 'development/assessment.json')):
        raise ValueError('protected gates require actual passing development of the frozen source')
    return contract, selected, campaign.record(freeze_path), assessment


def _directory(context, phase, gate_id=None):
    path = Path(context).resolve() / phase / 'protected'
    if gate_id is not None:
        if gate_id not in POLICY['gates']:
            raise ValueError('protected gate must be a or b')
        path /= 'gate-' + gate_id
    return path


def _producer_paths():
    from tools import compact_value_bfm_release_v2 as release
    return {name: Path(module.__file__) for name, module in (
        ('driver', sys.modules[__name__]), ('campaign', campaign), ('development', development),
        ('stream', stream), ('features', campaign.features), ('openings', campaign.openings),
        ('validator', gate), ('release', release), ('corpus', campaign.corpus),
        ('legacy', campaign.legacy), ('suite', development.suite), ('search', development.search),
        ('full_selection', development.full_selection), ('bootstrap', development.maintained),
        ('ci', release.ci), ('release_preflight', release.maintained))}


def _fingerprints(document):
    if document.get('domain') not in DOMAINS:
        raise ValueError('unknown protected exclusion fingerprint domain')
    values = document.get('fingerprints')
    if not isinstance(values, list) or any(not isinstance(value, str) or
            re.fullmatch('[0-9a-f]{64}', value) is None for value in values):
        raise ValueError('exclusion contains malformed fingerprints')
    return document['domain'], values


def _current_fingerprints(context, phase):
    positions = campaign.read(context / phase / 'positions.json')
    for binding in positions['census_files']:
        for row in stream.read_gzip(campaign.verify(binding)):
            yield from row['closure']
    games = campaign.read(context / phase / 'games.json')
    for row in games['rows']:
        actions = row['game']['transcript'].split('/')
        state = campaign.features.ReplayState()
        for turn, action in enumerate(actions):
            if turn >= row['game']['prefix_turns']:
                yield campaign.fingerprints(state)
            campaign.features.apply_complete_turn(state, state.to_move, action)
        yield campaign.fingerprints(state)


def _prepare_exclusions(context, phase, ready, *, create):
    """Freeze hashes only; never open any historical protected trajectory file."""
    contract, _, frozen, assessment = ready
    directory = _directory(context, phase)
    path = directory / 'exclusions.json'
    inputs = {'freeze': frozen, 'context': campaign.record(context / 'campaign.json'),
              'positions': campaign.record(context / phase / 'positions.json'),
              'games': campaign.record(context / phase / 'games.json'),
              'development': campaign.record(context / phase / 'development/assessment.json')}
    fragments = list(contract['exclusions']) + list(assessment['development_exclusions'])
    for binding in fragments:
        _fingerprints(campaign.read(campaign.verify(binding)))
    current_path = directory / 'current-fingerprints.jsonl.gz'
    # This re-exports only census hashes and unprotected played boundaries. The
    # deterministic writer verifies an existing export on independent validation.
    if not create and not current_path.exists():
        raise ValueError('protected exclusion export is missing')
    stream.write_gzip(current_path, _current_fingerprints(context, phase))
    return _seal_or_validate(path, {'schema': campaign.ID + '.protected-exclusions.v2',
        'inputs': inputs, 'fragments': fragments, 'current': campaign.record(current_path),
        'contains_transcripts': False, 'contains_labels': False, 'contains_metrics': False}, create=create)


def _collision_sets(exclusions, target, additional=()):
    collided = {domain: set() for domain in DOMAINS}
    for binding in exclusions['fragments']:
        domain, values = _fingerprints(campaign.read(campaign.verify(binding)))
        collided[domain].update(target[domain].intersection(values))
    for fps in stream.read_gzip(campaign.verify(exclusions['current'])):
        for domain, value in fps.items():
            if domain not in DOMAINS or re.fullmatch('[0-9a-f]{64}', value) is None:
                raise ValueError('current exclusion fingerprint changed')
            if value in target[domain]:
                collided[domain].add(value)
    for binding in additional:
        domain, values = _fingerprints(campaign.read(campaign.verify(binding)))
        collided[domain].update(target[domain].intersection(values))
    return collided


def _seed_claim(directory, gate_id, freeze, exclusions, producers, prior=None, *, create=True):
    path = directory / 'seed-claim.json'
    expected = {'schema': campaign.ID + '.protected-seed-claim.v2', 'gate': gate_id,
                'freeze': freeze, 'exclusions': exclusions, 'producers': producers,
                'previous_gate': prior, 'policy': POLICY}
    if path.exists():
        claim = campaign.read(path)
        if {key: value for key, value in claim.items() if key not in ('body_sha256', 'seed_hex')} != expected:
            raise ValueError('protected seed changed its frozen source or exclusion binding')
        if not isinstance(claim.get('seed_hex'), str) or re.fullmatch('[0-9a-f]{64}', claim['seed_hex']) is None:
            raise ValueError('protected seed is not 256 bits')
        if prior is not None:
            previous = campaign.read(campaign.verify(prior))
            previous_bank = campaign.read(campaign.verify(previous['bank']))
            previous_seed = campaign.read(campaign.verify(previous_bank['claim']))
            if claim['seed_hex'] == previous_seed['seed_hex']:
                raise ValueError('independent protected seeds collide')
        return claim
    if not create:
        raise ValueError('protected seed claim is missing')
    seed = os.urandom(32)
    if len(seed) != 32:
        raise ValueError('OS protected seed was not 256 bits')
    if prior is not None:
        previous = campaign.read(campaign.verify(prior))
        previous_bank = campaign.read(campaign.verify(previous['bank']))
        previous_seed = campaign.read(campaign.verify(previous_bank['claim']))
        if seed.hex() == previous_seed['seed_hex']:
            raise ValueError('independent protected seeds unexpectedly collide')
    return campaign.seal(path, {**expected, 'seed_hex': seed.hex()})


def _proposals(seed_hex):
    seed = bytes.fromhex(seed_hex)
    seen, rows = defaultdict(set), []
    for attempt in range(POLICY['proposal_attempt_limit']):
        generated = campaign.openings.generate_candidate(hashlib.sha256(seed + attempt.to_bytes(16, 'big')).digest())
        if generated is None:
            continue
        state, transcript, plies = generated
        fps = campaign.fingerprints(state)
        if any(value in seen[domain] for domain, value in fps.items()):
            continue
        for domain, value in fps.items():
            seen[domain].add(value)
        rows.append({'opening_id': fps[campaign.legacy.STATE_FINGERPRINT_DOMAIN],
                     'transcript': transcript, 'plies': plies, 'fingerprints': fps})
        if len(rows) == POLICY['proposal_roots']:
            return rows
    raise ValueError('protected proposal budget exhausted; seed cannot be replaced')


def _bank_tsv(rows):
    return ('opening_id\ttranscript\n' + ''.join(
        row['opening_id'] + '\t' + row['transcript'] + '\n' for row in rows)).encode('ascii')


def _prepare_bank(context, phase, gate_id, ready, exclusions, previous=None, *, create):
    directory = _directory(context, phase, gate_id)
    producers = {}
    for name, path in _producer_paths().items():
        destination = directory / 'provenance' / (name + '.py')
        if not create and not destination.exists():
            raise ValueError('protected producer evidence is missing')
        producers[name] = campaign.copy_checked(path, destination)
    previous_binding = campaign.record(_directory(context, phase, 'a') / 'assessment.json') if previous else None
    claim = _seed_claim(directory, gate_id, ready[2],
                        campaign.record(_directory(context, phase) / 'exclusions.json'), producers, previous_binding, create=create)
    proposal_path = directory / 'proposals.json'
    # Regeneration is source- and seed-bound, never informed by protected games.
    rows = _proposals(claim['seed_hex'])
    _seal_or_validate(proposal_path, {'schema': campaign.ID + '.protected-proposals.v2',
        'claim': campaign.record(directory / 'seed-claim.json'), 'rows': rows}, create=create)
    target = {domain: {row['fingerprints'][domain] for row in rows} for domain in DOMAINS}
    collided = _collision_sets(exclusions, target, previous['protected_exclusions'] if previous else ())
    retained = [row for row in rows if all(value not in collided[domain]
                for domain, value in row['fingerprints'].items())][:POLICY['pairs_per_gate']]
    development.validate_bank_rows(retained, pairs=500)
    if not create and not (directory / 'bank.tsv').exists():
        raise ValueError('protected bank TSV is missing')
    campaign.once(directory / 'bank.tsv', _bank_tsv(retained))
    gate.validate_bank(directory / 'bank.tsv')
    bank = _seal_or_validate(directory / 'bank.json', {'schema': campaign.ID + '.protected-bank.v2',
        'claim': campaign.record(directory / 'seed-claim.json'), 'tsv': campaign.record(directory / 'bank.tsv'),
        'proposals': campaign.record(proposal_path), 'rows': retained, 'pairs': 500, 'gate': gate_id}, create=create)
    return bank


def _compile_command(directory, compiler):
    return [str(compiler), '-std=c++20', '-O3',
        '-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="' + str(directory / 'candidate.cpp') + '"',
        str(directory / 'compact_value_bfm/rank4_gate_trajectories.cpp'), '-o', str(directory / 'gate.bin')]


def _build(directory, contract, selected):
    directory /= 'build'
    path = directory / 'build.json'
    if path.exists():
        return validate_build(campaign.record(path), contract, selected)
    sources = {'candidate': campaign.copy_checked(campaign.verify(selected['source']), directory / 'candidate.cpp'),
        'gate': campaign.copy_checked(GATE_SOURCE, directory / 'compact_value_bfm/rank4_gate_trajectories.cpp'),
        'rank4': campaign.copy_checked(campaign.verify(contract['opponents']['rank_4']['submission.cpp']),
                                      directory / 'rank_4/submission.cpp')}
    if sources['rank4']['sha256'] != gate.RANK4_SHA256:
        raise ValueError('protected gate changed historical Rank4')
    compiler = campaign.verify(contract['compiler'])
    command = _compile_command(directory, compiler)
    with (directory / 'build.log').open('xb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, **campaign.THREADS})
    campaign.seal(path, {'schema': campaign.ID + '.protected-build.v2',
        'candidate': selected['source'], 'runtime': selected['runtime'], 'sources': sources,
        'gate_original': campaign.record(GATE_SOURCE), 'compiler': contract['compiler'],
        'command': command, 'binary': campaign.record(directory / 'gate.bin'),
        'log': campaign.record(directory / 'build.log'), 'policy': POLICY})
    return validate_build(campaign.record(path), contract, selected)


def validate_build(binding, contract, selected):
    path = campaign.verify(binding)
    build, directory = campaign.read(path), path.parent
    expected = {'candidate': directory / 'candidate.cpp',
                'gate': directory / 'compact_value_bfm/rank4_gate_trajectories.cpp',
                'rank4': directory / 'rank_4/submission.cpp'}
    if (path.name != 'build.json' or build['candidate'] != selected['source']
            or build['runtime'] != selected['runtime'] or build['compiler'] != contract['compiler']
            or build['policy'] != POLICY or build['gate_original'] != campaign.record(GATE_SOURCE)
            or set(build['sources']) != set(expected)
            or campaign.verify(build['binary']) != directory / 'gate.bin'
            or campaign.verify(build['log']) != directory / 'build.log'):
        raise ValueError('protected native build source, runtime, binary or compiler changed')
    for name, original in (('candidate', selected['source']), ('gate', build['gate_original']),
                           ('rank4', contract['opponents']['rank_4']['submission.cpp'])):
        campaign.verify(original)
        if (campaign.verify(build['sources'][name]) != expected[name]
                or any(build['sources'][name][key] != original[key] for key in ('sha256', 'bytes'))):
            raise ValueError('protected compiled source copy differs from its frozen original')
    if (build['sources']['rank4']['sha256'] != gate.RANK4_SHA256
            or build['command'] != _compile_command(directory, campaign.verify(build['compiler']))):
        raise ValueError('protected build command or historical Rank4 changed')
    return build


def shard_command(directory, ordinal, bank, build):
    if type(ordinal) is not int or not 0 <= ordinal < 100:
        raise ValueError('protected shard ordinal must be in [0,100)')
    return [build['binary']['path'], '--bank', bank['tsv']['path'],
        '--candidate-source', build['sources']['candidate']['path'],
        '--rank4-source', build['sources']['rank4']['path'],
        '--output', str(directory / 'shards' / f'{ordinal:03d}' / 'result.json'),
        '--expected-bank-sha256', bank['tsv']['sha256'],
        '--expected-candidate-sha256', build['candidate']['sha256'],
        '--pair-offset', str(ordinal * 5), '--pair-count', '5', '--mode', 'actual-clock',
        '--max-turns', '320', '--include-trajectories']


def _shard_expected(directory, ordinal, execution):
    return {'schema': campaign.ID + '.protected-shard-claim.v2', 'ordinal': ordinal,
            'execution': campaign.record(directory / 'execution-claim.json'),
            'command': execution['commands'][ordinal], 'retry_allowed': False}


def validate_shard(directory, ordinal, execution, bank, build, selected):
    output = directory / 'shards' / f'{ordinal:03d}'
    claim_path, receipt_path = output / 'claim.json', output / 'receipt.json'
    if not claim_path.exists():
        raise ValueError('protected shard has no start claim')
    claim = campaign.read(claim_path)
    if {key: value for key, value in claim.items() if key != 'body_sha256'} != _shard_expected(directory, ordinal, execution):
        raise ValueError('protected shard claim belongs to another execution')
    try:
        receipt = campaign.read(receipt_path)
        if (receipt['claim'] != campaign.record(claim_path) or receipt['returncode'] not in (0, 2)
                or receipt['timeout'] or campaign.verify(receipt['raw']) != output / 'result.json'):
            raise ValueError('incomplete protected execution')
        for key, filename in (('stdout', 'stdout.log'), ('stderr', 'stderr.log')):
            if campaign.verify(receipt[key]) != output / filename:
                raise ValueError('protected shard log belongs to another execution')
        if receipt.get('trajectory_progress'):
            if campaign.verify(receipt['trajectory_progress']) != output / 'result.json.trajectories.jsonl':
                raise ValueError('protected shard progress belongs to another execution')
        checked = gate.validate_result(campaign.verify(receipt['raw']),
            expected_bank_sha256=bank['tsv']['sha256'], expected_candidate_sha256=selected['source']['sha256'],
            expected_candidate_search_profile=selected['candidate_search_profile'], require_trajectories=True,
            trajectory_bank=campaign.verify(bank['tsv']))
        config, bindings = checked['config'], checked['bindings']
        if (config['mode'] != 'actual-clock' or config['pair_offset'] != ordinal * 5 or config['pair_count'] != 5
                or config['max_turns'] != 320 or config['minimum_candidate_wins'] != -1
                or config['minimum_wins_per_color'] != -1
                or config['candidate_clocks_ms'] != [800, 155] or config['rank4_clocks_ms'] != [800, 165]
                or bindings['candidate_runtime_body_sha256'] != selected['runtime_body_sha256']
                or bindings['candidate_payload_sha256'] != selected['payload_sha256']
                or bindings['candidate_source_bytes'] != build['sources']['candidate']['bytes']
                or bindings['rank4_source_bytes'] != build['sources']['rank4']['bytes']
                or receipt['returncode'] != (0 if checked['result']['passed'] else 2)):
            raise ValueError('protected shard configuration or compiled model changed')
        expected = [(index, bank['rows'][index]['opening_id'], color)
                    for index in range(ordinal * 5, ordinal * 5 + 5) for color in (0, 1)]
        if [(row['pair_index'], row['opening_id'], row['candidate_player']) for row in checked['games']] != expected:
            raise ValueError('protected shard schedule changed')
        return checked
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise SpentShardError(f'protected shard {ordinal} is spent without a valid completion: {error}') from error


def _execute_shard(directory, ordinal, execution, bank, build, selected):
    output = directory / 'shards' / f'{ordinal:03d}'
    if (output / 'claim.json').exists():
        return validate_shard(directory, ordinal, execution, bank, build, selected)
    if output.exists() and any(output.iterdir()):
        raise SpentShardError(f'protected shard {ordinal} has orphaned execution evidence')
    command = shard_command(directory, ordinal, bank, build)
    if command != execution['commands'][ordinal]:
        raise ValueError('protected worker command differs from frozen execution')
    campaign.verify(build['binary']); campaign.verify(bank['tsv'])
    for binding in build['sources'].values():
        campaign.verify(binding)
    campaign.seal(output / 'claim.json', _shard_expected(directory, ordinal, execution))
    started = time.monotonic()
    with (output / 'stdout.log').open('xb') as out, (output / 'stderr.log').open('xb') as err:
        try:
            process = subprocess.run(command, stdout=out, stderr=err,
                env={**os.environ, **campaign.THREADS}, timeout=POLICY['shard_watchdog_seconds'])
            returncode, timeout = process.returncode, False
        except subprocess.TimeoutExpired:
            returncode, timeout = None, True
    result, progress = output / 'result.json', output / 'result.json.trajectories.jsonl'
    campaign.seal(output / 'receipt.json', {'schema': campaign.ID + '.protected-shard-receipt.v2',
        'claim': campaign.record(output / 'claim.json'), 'returncode': returncode, 'timeout': timeout,
        'elapsed_seconds': time.monotonic() - started,
        'raw': campaign.record(result) if result.exists() else None,
        'trajectory_progress': campaign.record(progress) if progress.exists() else None,
        'stdout': campaign.record(output / 'stdout.log'), 'stderr': campaign.record(output / 'stderr.log')})
    return validate_shard(directory, ordinal, execution, bank, build, selected)


def assess_games(games, rows):
    expected = [(index, row['opening_id'], color) for index, row in enumerate(rows) for color in (0, 1)]
    if len(rows) != 500 or [(row['pair_index'], row['opening_id'], row['candidate_player']) for row in games] != expected:
        raise ValueError('protected aggregate requires exactly 500 canonical pairs and both colors')
    wins = [sum(row['failure'] is None and row['winner'] == color
                for row in games if row['candidate_player'] == color) for color in (0, 1)]
    failures = sum(row['failure'] is not None for row in games)
    return {'games': 1000, 'pairs': 500, 'candidate_wins': sum(wins),
            'candidate_wins_by_color': wins, 'failures': failures,
            'passed': sum(wins) >= 527 and min(wins) >= 260 and failures == 0}


def _execution_claim(directory, ready, bank, build, *, create):
    contract, selected, frozen, _ = ready
    seed = campaign.read(directory / 'seed-claim.json')
    expected = {'schema': campaign.ID + '.protected-execution-claim.v2',
        'freeze': frozen, 'selected': selected, 'bank': campaign.record(directory / 'bank.json'),
        'build': campaign.record(directory / 'build/build.json'), 'producers': seed['producers'],
        'commands': [shard_command(directory, index, bank, build) for index in range(100)],
        'policy': POLICY, 'threads': campaign.THREADS, 'nice': 0}
    for name, path in _producer_paths().items():
        original = campaign.record(path)
        if any(seed['producers'][name][key] != original[key] for key in ('sha256', 'bytes')):
            raise ValueError('protected producers changed; use the frozen source snapshot')
        campaign.verify(seed['producers'][name])
    path = directory / 'execution-claim.json'
    if create and not path.exists():
        campaign.seal(path, expected)
    actual = campaign.read(path)
    if {key: value for key, value in actual.items() if key != 'body_sha256'} != expected:
        raise ValueError('protected execution changed frozen source, binary, bank or command')
    return actual


def _finalize_gate(context, phase, gate_id, ready, bank, build, execution):
    directory = _directory(context, phase, gate_id)
    games, receipts, configuration = [], [], None
    for ordinal in range(100):
        checked = validate_shard(directory, ordinal, execution, bank, build, ready[1])
        config = {key: value for key, value in checked['config'].items() if key != 'pair_offset'}
        if configuration is not None and config != configuration:
            raise ValueError('protected shards used inconsistent search configurations')
        configuration = config
        games.extend(checked['games'])
        receipts.append(campaign.record(directory / 'shards' / f'{ordinal:03d}' / 'receipt.json'))
    summary = assess_games(games, bank['rows'])
    path = directory / 'execution.json'
    campaign.seal(path, {'schema': campaign.ID + '.protected-execution.v2',
        'claim': campaign.record(directory / 'execution-claim.json'), 'shards': receipts, 'result': summary})
    values = defaultdict(set)
    # Every materialized proposal is protected, including unused bank proposals.
    for row in campaign.read(campaign.verify(bank['proposals']))['rows']:
        for domain, value in row['fingerprints'].items():
            values[domain].add(value)
    for game in games:
        for fps in development.boundaries(game['transcript'], game['root_transcript']):
            for domain, value in fps.items():
                values[domain].add(value)
    exclusions = []
    for ordinal, domain in enumerate(DOMAINS):
        exclusion = directory / f'protected-exclusion-{ordinal}.json'
        campaign.seal(exclusion, {'schema': campaign.ID + '.protected-played-exclusions.v2',
            'role': 'protected', 'domain': domain, 'fingerprints': sorted(values[domain]),
            'execution': campaign.record(path), 'bank': campaign.record(directory / 'bank.json'),
            'contains_transcripts': False, 'contains_labels': False, 'contains_metrics': False,
            'includes_all_proposals': True, 'includes_all_played_postroot_boundaries': True,
            'includes_terminal_features': True})
        exclusions.append(campaign.record(exclusion))
    return campaign.seal(directory / 'assessment.json', {'schema': campaign.ID + '.protected-assessment.v2',
        'gate': gate_id, 'freeze': ready[2], 'selected': ready[1],
        'bank': campaign.record(directory / 'bank.json'), 'execution': campaign.record(path),
        'protected_exclusions': exclusions, 'policy': POLICY, **summary, 'campaign_success': False})


def _process(root, context, phase, *, execute):
    ready = prerequisites(root, context, phase)
    exclusions = _prepare_exclusions(context, phase, ready, create=execute)
    previous, results = None, []
    for gate_id in POLICY['gates']:
        directory = _directory(context, phase, gate_id)
        if not execute and not (directory / 'assessment.json').exists():
            raise ValueError('protected validation requires both completed assessments')
        if not execute and not (directory / 'bank.json').exists():
            raise ValueError('protected validator cannot materialize a bank')
        if execute:
            build = _build(directory, ready[0], ready[1])
        else:
            build = validate_build(campaign.record(directory / 'build/build.json'), ready[0], ready[1])
        bank = _prepare_bank(context, phase, gate_id, ready, exclusions, previous, create=execute)
        execution = _execution_claim(directory, ready, bank, build, create=execute)
        pending = []
        # Validate every existing claim before launching anything. One spent
        # shard prevents subsequent untouched shards from consuming more roots.
        for ordinal in range(100):
            output = directory / 'shards' / f'{ordinal:03d}'
            if (output / 'claim.json').exists():
                validate_shard(directory, ordinal, execution, bank, build, ready[1])
            elif output.exists() and any(output.iterdir()):
                raise SpentShardError(f'protected shard {ordinal} has orphaned evidence')
            else:
                pending.append(ordinal)
        if pending:
            if not execute:
                raise ValueError('protected assessment has missing shards')
            if os.getpriority(os.PRIO_PROCESS, 0) != 0:
                raise ValueError('actual-clock protected execution requires nice zero')
            # Four workers claim their own next shard only when ready to run it.
            # No claims are pre-issued for pending work.
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                for offset in range(0, len(pending), 4):
                    futures = [pool.submit(_execute_shard, directory, ordinal, execution, bank, build, ready[1])
                               for ordinal in pending[offset:offset + 4]]
                    for future in futures:
                        future.result()
        was_complete = (directory / 'assessment.json').exists()
        previous = _finalize_gate(context, phase, gate_id, ready, bank, build, execution)
        results.append(previous)
        if execute and not was_complete:
            campaign.event(Path(root), 'protected-gate-completed', {'gate': gate_id,
                'assessment': campaign.record(directory / 'assessment.json'), 'passed': previous['passed'],
                'campaign_success': False})
    passed = all(row['passed'] for row in results)
    return campaign.seal(_directory(context, phase) / 'assessment.json', {
        'schema': campaign.ID + '.dual-protected-assessment.v2', 'freeze': ready[2],
        'selected': ready[1], 'gates': [campaign.record(_directory(context, phase, key) / 'assessment.json')
                                     for key in POLICY['gates']],
        'protected_exclusions': [binding for row in results for binding in row['protected_exclusions']],
        'policy': POLICY, 'passed': passed,
        'status': 'protected-passed-awaiting-source-bound-live' if passed else 'protected-rejected',
        'campaign_success': False})


def run(root, context, phase):
    root, context = Path(root).resolve(), Path(context).resolve()
    with campaign.lease(root):
        return _process(root, context, phase, execute=True)


def validate(root, context, phase):
    """Reproduce both gates, without execution, fresh seeds or unclaimed banks."""
    root, context = Path(root).resolve(), Path(context).resolve()
    if not (_directory(context, phase) / 'assessment.json').exists():
        raise ValueError('protected validation requires a completed dual assessment')
    return _process(root, context, phase, execute=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('command', choices=('run', 'validate'))
    args = parser.parse_args()
    result = (run if args.command == 'run' else validate)(args.root, args.context, args.phase)
    print(json.dumps({key: result[key] for key in ('status', 'passed', 'campaign_success')}), flush=True)


if __name__ == '__main__':
    main()
