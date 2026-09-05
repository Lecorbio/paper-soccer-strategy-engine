#!/usr/bin/env python3
"""Execute frozen, paired state-capable six-opponent qualification suites."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_opponent_evaluation as evaluation
from tools import compact_value_bfm_stream_v2 as stream

PAIRS = {'screen': 32, 'confirmation': 100}
DEPTHS = (8, 12, 20, 40)
OPPONENT_CLOCKS = {name: ((800, 165) if name in campaign.OPPONENTS[:3] else
    (800, 155) if name == 'jacek_native_bfm' else (650, 130)) for name in campaign.OPPONENTS}
POLICY = {'schema': campaign.ID + '.opponent-suite-policy.v2',
    'pairs_per_opponent': PAIRS, 'root_depths_equal': list(DEPTHS),
    'proposal_multiplier': 8, 'candidate_clocks_ms': [800, 155],
    'opponent_clocks_ms': {name: list(value) for name, value in OPPONENT_CLOCKS.items()},
    'external_deadlines_ms': [1000, 200], 'workers': 4,
    'pair_watchdog_seconds': 180, 'retry_claimed_suite': False,
    'control': 'frozen-deployed-upload', 'bootstrap_replicates': 10000,
    'improvement': .03, 'paired_lower_above': 0, 'maximum_opponent_regression': .05,
    'adapter_emergency_in_search_budget': True, 'adapter_search_exception_is_failure': True,
    'empty_games_require_process_referee': True, 'proxy_opponents': True}


def source_selection(context, phase):
    from tools import compact_value_bfm_search_v2 as search
    from tools import compact_value_bfm_train as trainer
    context=Path(context).resolve()
    contract=campaign.read(context/'campaign.json')
    root=campaign.verify(contract['parent_campaign']).parent
    with trainer.native_thread_execution_scope():
        selected=search.validate_selection(root,context,phase)
    if selected.get('required_ablation_complete') is not True or selected.get('eligible_for_multi_opponent') is not True:
        raise ValueError('opponent suite requires completed source-bound search selection')
    return selected,search.directory(context,phase)/'search-selection.json'


def search_boundaries(selection):
    """Every played search variant remains excluded, including rejected arms."""
    strength=campaign.read(campaign.verify(selection['strength']))
    for record in strength['executions'].values():
        execution=campaign.read(campaign.verify(record))
        raw=json.loads(campaign.verify(execution['raw']).read_bytes())
        for game in raw['games']:
            state=campaign.features.ReplayState();prefix=len(game['root_transcript'].split('/'))
            for turn,action in enumerate(game['transcript'].split('/')):
                if turn>=prefix:yield campaign.fingerprints(state)
                campaign.features.apply_complete_turn(state,state.to_move,action)
            yield campaign.fingerprints(state)


def replay(transcript):
    state = campaign.features.ReplayState()
    for action in transcript.split('/'):
        if not action:
            raise ValueError('empty or malformed root/trajectory')
        campaign.features.apply_complete_turn(state, state.to_move, action)
    return state


def validate_bank_rows(rows, pairs):
    """Check the actual canonical states, independently of human-readable IDs."""
    if set(rows) != set(campaign.OPPONENTS):
        raise ValueError('incomplete frozen opponent bank')
    seen_ids = set()
    seen = {}
    for name in campaign.OPPONENTS:
        if len(rows[name]) != pairs:
            raise ValueError('wrong stage root count')
        depths = []
        for row in rows[name]:
            state = replay(row['transcript'])
            fps = campaign.fingerprints(state)
            edges = len(state.used_segments)
            if state.winner is not None or edges != row['edges'] or edges not in DEPTHS:
                raise ValueError('bank progress or terminal state changed')
            if fps != row['fingerprints']:
                raise ValueError('bank canonical fingerprints changed')
            if row['root_id'] in seen_ids:
                raise ValueError('duplicate root ID')
            seen_ids.add(row['root_id'])
            for domain, value in fps.items():
                if value in seen.setdefault(domain, set()):
                    raise ValueError('canonical root reused within/across opponents')
                seen[domain].add(value)
            depths.append(edges)
        if any(depths.count(depth) != pairs // 4 for depth in DEPTHS):
            raise ValueError('root depth marginals changed')


def checked_games(path, root, opponent):
    """Reconcile native evidence against the frozen root and legal trajectory."""
    games = evaluation.load_games(path)
    if set(games) != {(root['root_id'], color) for color in (0, 1)}:
        raise ValueError('native root/color schedule differs from frozen bank')
    for (_, color), row in games.items():
        if (row['root_transcript'] != root['transcript'] or row['root_edges'] != root['edges']
                or [row['first_budget_ms'], row['later_budget_ms']] != [800, 155]
                or (row['opponent_first_budget_ms'], row['opponent_later_budget_ms']) != OPPONENT_CLOCKS[opponent]):
            raise ValueError('native root or actor clocks changed')
        parts = row['trajectory'].split('/')
        prefix = root['transcript'].split('/')
        if parts[:len(prefix)] != prefix or len(parts) - len(prefix) != row['turns']:
            raise ValueError('native trajectory lost its complete-turn prefix')
        state = replay(row['trajectory'])
        if row['winner'] != (state.winner if state.winner is not None else -1):
            raise ValueError('native winner differs from legal replay')
        if not row['failure'] and state.winner is None:
            raise ValueError('unfinished native game reported as successful')
        if not isinstance(row['turns'], int) or not 0 <= row['turns'] <= 320:
            raise ValueError('invalid native turn count')
        latencies = row['candidate_latency_ms']
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in latencies):
            raise ValueError('invalid native latency evidence')
        if row['candidate_max_ms'] != max(latencies, default=0):
            raise ValueError('native latency maximum differs')
    return games


def _completed_suite(directory):
    directory=Path(directory).resolve()
    result = campaign.read(directory / 'assessment.json')
    if Path(result['manifest']['path']) != directory / 'manifest.json':
        raise ValueError('suite execution manifest belongs to another stage')
    manifest = campaign.read(campaign.verify(result['manifest']))
    claim = campaign.read(campaign.verify(manifest['claim']))
    if (claim['bank'] != manifest['bank'] or claim['builds'] != manifest['builds']
            or claim['policy'] != POLICY):
        raise ValueError('suite manifest differs from execution claim')
    if (Path(manifest['claim']['path']) != directory / 'execution-claim.json'
            or Path(manifest['bank']['path']) != directory / 'bank.json'):
        raise ValueError('suite claim or bank belongs to another stage')
    campaign.verify(claim['driver']); campaign.verify(claim['assessment_driver']); campaign.verify(claim['adapter_source'])
    bank = campaign.read(campaign.verify(manifest['bank']))
    seed = campaign.read(campaign.verify(bank['claim']))
    context=directory.parents[2];phase=directory.parents[1].name
    expected_inputs={'context':campaign.record(context/'campaign.json'),
        'selection':campaign.record(context/phase/'search/search-selection.json'),
        'full_model_selection':campaign.record(context/phase/'full-model-selection.json'),
        'positions':campaign.record(context/phase/'positions.json'),'games':campaign.record(context/phase/'games.json')}
    if directory.name=='confirmation':expected_inputs['screen']=campaign.record(directory.parent/'screen/assessment.json')
    if seed['inputs']!=expected_inputs or Path(bank['claim']['path'])!=directory/'seed-claim.json':
        raise ValueError('suite seed inputs belong to another phase')
    if (bank['stage'] != directory.name or bank['stage'] != seed['stage'] or seed['policy'] != POLICY
            or bank['selection'] != claim['selection'] or bank['selection'] != seed['inputs']['selection']):
        raise ValueError('suite stage or selection claim changed')
    for binding in seed['inputs'].values(): campaign.verify(binding)
    contract = campaign.read(campaign.verify(seed['inputs']['context']))
    parent = campaign.read(campaign.verify(contract['parent_campaign']))
    selection = campaign.read(campaign.verify(claim['selection']))
    validated,selection_path=source_selection(context,phase)
    if selection!=validated or claim['selection']!=campaign.record(selection_path) or selection['full_model_selection']!=expected_inputs['full_model_selection']:
        raise ValueError('suite source is not the measured and strength-tested search selection')
    expected_sources = {'candidate': selection['selected']['source'], 'control': parent['inputs']['discrete_v3_deployment.cpp']}
    if selection.get('eligible_for_multi_opponent') is not True or claim['sources'] != expected_sources:
        raise ValueError('suite candidate or frozen deployed control changed')
    validate_bank_rows(bank['rows'], PAIRS[bank['stage']])
    validate_bank_isolation(context,phase,bank,selection)
    arms = {arm: {name: {} for name in campaign.OPPONENTS} for arm in ('candidate', 'control')}
    if set(manifest['builds']) != {arm + ':' + name for arm in arms for name in campaign.OPPONENTS}:
        raise ValueError('suite adapter roster changed')
    for arm in arms:
        for name in campaign.OPPONENTS:
            validate_build(manifest['builds'][arm + ':' + name], contract, expected_sources[arm], name, claim['adapter_source'])
    for name, rows in bank['rows'].items():
        for row in rows:
            path = campaign.verify(bank['tsvs'][name + ':' + row['root_id']])
            if path.read_text() != 'root_id\ttranscript\n' + row['root_id'] + '\t' + row['transcript'] + '\n':
                raise ValueError('native TSV differs from frozen canonical bank')
    if len(manifest['pairs']) != 6 * PAIRS[bank['stage']]:
        raise ValueError('suite has missing or duplicated pair evidence')
    expected = {(name, row['root_id']): row for name, rows in bank['rows'].items() for row in rows}
    seen = set()
    for record in manifest['pairs']:
        receipt = campaign.read(campaign.verify(record))
        key = (receipt['opponent'], receipt['root']['root_id'])
        if key in seen or expected.get(key) != receipt['root']:
            raise ValueError('suite pair differs from frozen schedule')
        seen.add(key)
        pair_directory=directory/'games'/receipt['opponent']/receipt['root']['root_id']
        if Path(record['path'])!=pair_directory/'pair.json':
            raise ValueError('suite pair belongs to another execution')
        for arm in arms:
            execution = receipt['arms'][arm]
            expected_build = manifest['builds'][arm + ':' + receipt['opponent']]
            build = campaign.read(campaign.verify(expected_build))
            expected_command = [build['binary']['path'], bank['tsvs'][receipt['opponent'] + ':' + receipt['root']['root_id']]['path'], '800', '155']
            if execution['build'] != expected_build or execution['command'] != expected_command:
                raise ValueError('native pair ran a different source, root or clock command')
            if Path(execution['output']['path'])!=pair_directory/(arm+'.jsonl') or Path(execution['stderr']['path'])!=pair_directory/(arm+'.stderr'):
                raise ValueError('native pair output belongs to another execution')
            if execution['returncode'] != 0 or execution['timeout']:
                raise ValueError('suite has an incomplete native process')
            output = campaign.verify(execution['output'])
            campaign.verify(execution['stderr'])
            arms[arm][receipt['opponent']].update(checked_games(output, receipt['root'], receipt['opponent']))
    reproduced = evaluation.assess(arms['candidate'], arms['control'])
    if any(result.get(key) != value for key, value in reproduced.items()):
        raise ValueError('suite assessment does not reproduce native evidence')
    return result, bank, manifest


def _current_collisions(context, phase, target):
    collided = {domain: set() for domain in target}
    def add(fps):
        for domain, value in fps.items():
            if value in target[domain]:
                collided[domain].add(value)
    positions = campaign.read(context / phase / 'positions.json')
    for record in positions['census_files']:
        for row in stream.read_gzip(campaign.verify(record)):
            for member in row['closure']:
                add(member)
    games = campaign.read(context / phase / 'games.json')
    for row in games['rows']:
        state = campaign.features.ReplayState()
        for turn, action in enumerate(row['game']['transcript'].split('/')):
            if turn >= row['game']['prefix_turns']:
                add(campaign.fingerprints(state))
            campaign.features.apply_complete_turn(state, state.to_move, action)
        add(campaign.fingerprints(state))
    return collided


def validate_bank_isolation(context,phase,bank,selection,previous=None):
    """Recheck exact retained states against immutable corpus and played games."""
    target={}
    excluded=campaign.exclusion_sets(campaign.read(Path(context)/'campaign.json'))
    for rows in bank['rows'].values():
        for row in rows:
            if campaign.rejection(replay(row['transcript']),'validation',excluded):
                raise ValueError('suite root overlaps an excluded prior state')
            for domain,value in row['fingerprints'].items():target.setdefault(domain,set()).add(value)
    collided=_current_collisions(Path(context),phase,target)
    for fps in search_boundaries(selection):
        for domain,value in fps.items():
            if value in target[domain]:collided[domain].add(value)
    if bank['stage']=='confirmation':
        previous=previous or _completed_suite(Path(context)/phase/'multi-opponent/screen')
        if not previous[0]['passed'] or previous[1]['selection']!=bank['selection']:
            raise ValueError('confirmation bank changed its screened source')
        for record in previous[2]['pairs']:
            receipt=campaign.read(campaign.verify(record))
            for arm in ('candidate','control'):
                for game in evaluation.load_games(campaign.verify(receipt['arms'][arm]['output'])).values():
                    state=campaign.features.ReplayState();prefix=len(game['root_transcript'].split('/'))
                    for turn,action in enumerate(game['trajectory'].split('/')):
                        if turn>=prefix:
                            for domain,value in campaign.fingerprints(state).items():
                                if value in target[domain]:collided[domain].add(value)
                        campaign.features.apply_complete_turn(state,state.to_move,action)
                    for domain,value in campaign.fingerprints(state).items():
                        if value in target[domain]:collided[domain].add(value)
    if any(collided.values()):raise ValueError('suite root overlaps current corpus or played evaluation states')
    return True


def prepare_bank(context, phase, stage, selection_path, prior=None):
    context=Path(context).resolve();selection_path=Path(selection_path).resolve()
    directory = context / phase / 'multi-opponent' / stage
    selection = campaign.read(selection_path)
    validated,expected_selection_path=source_selection(context,phase)
    if selection!=validated or selection_path!=expected_selection_path:
        raise ValueError('suite root seed precedes validated source selection')
    if selection.get('eligible_for_multi_opponent') is not True or not selection.get('selected'):
        raise ValueError('full candidate must be selected before suite roots')
    selected = selection['selected']
    campaign.verify(selected['source']); campaign.verify(selected['runtime'])
    contract = campaign.read(context / 'campaign.json')
    inputs = {'selection': campaign.record(selection_path), 'context': campaign.record(context / 'campaign.json'),
        'full_model_selection':selection['full_model_selection'],
        'positions': campaign.record(context / phase / 'positions.json'), 'games': campaign.record(context / phase / 'games.json')}
    previous = None
    if stage == 'confirmation':
        previous = _completed_suite(directory.parent / 'screen')
        if not previous[0]['passed']:
            raise ValueError('confirmation requires a passing source-bound screen')
        prior = directory.parent / 'screen' / 'assessment.json'
        screen_selection = previous[1]['selection']
        if screen_selection != inputs['selection']:
            raise ValueError('confirmation candidate differs from screened selection')
        inputs['screen'] = campaign.record(prior)
    elif prior is not None:
        raise ValueError('unexpected previous evaluation')
    pairs = PAIRS[stage]
    claim = campaign.seal(directory / 'seed-claim.json', {'schema': campaign.ID + '.suite-seed-claim.v2',
        'inputs': inputs, 'stage': stage, 'policy': POLICY})
    if (directory / 'bank.json').exists():
        bank = campaign.read(directory / 'bank.json')
        if bank['claim'] != campaign.record(directory / 'seed-claim.json'):
            raise ValueError('resumed suite bank inputs changed')
        validate_bank_rows(bank['rows'], pairs)
        for item in bank['tsvs'].values(): campaign.verify(item)
        validate_bank_isolation(context,phase,bank,selection,previous)
        return bank
    excluded = campaign.exclusion_sets(contract)
    seen = {}; proposals = {name: [] for name in campaign.OPPONENTS}
    for name in campaign.OPPONENTS:
        for depth in DEPTHS:
            seed = hashlib.sha256(campaign.raw([claim['body_sha256'], name, depth])).digest()
            rng = random.Random(seed)
            pool = []
            for _ in range(100000):
                state, transcript = campaign.fresh_root(depth, rng)
                fps = campaign.fingerprints(state)
                if campaign.rejection(state, 'validation', excluded) or any(value in seen.get(domain, set()) for domain, value in fps.items()):
                    continue
                for domain, value in fps.items(): seen.setdefault(domain, set()).add(value)
                pool.append({'transcript': transcript, 'edges': depth, 'fingerprints': fps})
                if len(pool) == pairs // 4 * POLICY['proposal_multiplier']: break
            else:
                raise ValueError('frozen suite root proposal budget exhausted')
            proposals[name].extend(pool)
    target = {domain: set(values) for domain, values in seen.items()}
    collided = _current_collisions(context, phase, target)
    for fps in search_boundaries(selection):
        for domain,value in fps.items():
            if value in target[domain]:collided[domain].add(value)
    if previous:
        # Fresh confirmation excludes both screen roots and all played boundaries.
        for record in previous[2]['pairs']:
            receipt = campaign.read(campaign.verify(record))
            for arm in ('candidate', 'control'):
                for game in evaluation.load_games(campaign.verify(receipt['arms'][arm]['output'])).values():
                    state = campaign.features.ReplayState()
                    prefix_turns = len(game['root_transcript'].split('/'))
                    for turn, action in enumerate(game['trajectory'].split('/')):
                        if turn >= prefix_turns:
                            for domain, value in campaign.fingerprints(state).items():
                                if value in target[domain]: collided[domain].add(value)
                        campaign.features.apply_complete_turn(state, state.to_move, action)
                    for domain, value in campaign.fingerprints(state).items():
                        if value in target[domain]: collided[domain].add(value)
    rows = {}; tsvs = {}
    for name in campaign.OPPONENTS:
        retained = []
        for depth in DEPTHS:
            pool = [row for row in proposals[name] if row['edges'] == depth and
                all(value not in collided[domain] for domain, value in row['fingerprints'].items())]
            if len(pool) < pairs // 4:
                raise ValueError('insufficient isolated roots in frozen suite proposal pool')
            retained.extend(pool[:pairs // 4])
        for row in retained:
            row['root_id'] = row['fingerprints'][campaign.legacy.STATE_FINGERPRINT_DOMAIN]
        rows[name] = retained
        for row in retained:
            path = directory / 'banks' / name / (row['root_id'] + '.tsv')
            campaign.once(path, ('root_id\ttranscript\n' + row['root_id'] + '\t' + row['transcript'] + '\n').encode())
            tsvs[name + ':' + row['root_id']] = campaign.record(path)
    validate_bank_rows(rows, pairs)
    return campaign.seal(directory / 'bank.json', {'schema': campaign.ID + '.opponent-suite-bank.v2',
        'stage': stage, 'selection': inputs['selection'], 'claim': campaign.record(directory / 'seed-claim.json'),
        'rows': rows, 'tsvs': tsvs, 'pairs_per_opponent': pairs, 'fresh_from_screen': stage == 'confirmation'})


def compile_command(output, compiler, name):
    command = [str(compiler), '-std=c++20', '-O3', '-I' + str(output / 'include'), '-I' + str(output / 'src/bots'),
        '-DCAMPAIGN_CANDIDATE_SOURCE="' + str(output / 'candidate.cpp') + '"',
        '-DCAMPAIGN_OPPONENT_SOURCE="' + str(output / 'opponent/bot.cpp') + '"']
    if name in campaign.OPPONENTS[:4]: command.append('-DCAMPAIGN_OPPONENT_REPLAY_CORRECTION')
    return command + [str(output / 'adapter.cpp'), str(output / 'src/core/geometry.cpp'), str(output / 'src/core/rules.cpp'), '-o', str(output / 'adapter.bin')]


def shared_paths():
    return sorted((campaign.REPO / 'include').rglob('*.hpp')) + [campaign.REPO / 'src/bots/mcts_internal.hpp',
        campaign.REPO / 'src/core/geometry.cpp', campaign.REPO / 'src/core/rules.cpp']


def validate_build(binding, contract, candidate, name, adapter):
    manifest_path = campaign.verify(binding)
    output = manifest_path.parent
    if manifest_path.name != 'build.json':
        raise ValueError('adapter build receipt path changed')
    result = campaign.read(manifest_path)
    if Path(result['binary']['path']) != output / 'adapter.bin':
        raise ValueError('executed binary differs from compiler output')
    for key, source in result['sources'].items():
        expected = output / ({'candidate': 'candidate.cpp', 'adapter': 'adapter.cpp'}.get(key, key))
        if Path(source['path']) != expected:
            raise ValueError('source record differs from compiler input path')
    campaign.verify(candidate); campaign.verify(result['binary']); campaign.verify(result['compiler'])
    for item in result['sources'].values(): campaign.verify(item)
    if (result['candidate'] != candidate or result['opponent'] != contract['opponents'][name]
            or result['compiler'] != contract['compiler'] or result['policy'] != POLICY):
        raise ValueError('compiled adapter source identity or policy changed')
    for field in ('sha256', 'bytes'):
        if result['sources']['candidate'][field] != candidate[field]:
            raise ValueError('compiled candidate copy differs')
        if result['sources']['adapter'][field] != adapter[field]:
            raise ValueError('compiled adapter copy differs')
        for filename, original in contract['opponents'][name].items():
            if result['sources']['opponent/' + filename][field] != original[field]:
                raise ValueError('compiled opponent copy differs')
    expected_keys = {'candidate', 'adapter'} | {'opponent/' + filename for filename in contract['opponents'][name]}
    for path in shared_paths():
        route = str(path.relative_to(campaign.REPO)); expected_keys.add(route)
        original = subprocess.check_output(['git', 'show', campaign.START + ':' + route], cwd=campaign.REPO)
        if result['sources'][route]['sha256'] != hashlib.sha256(original).hexdigest():
            raise ValueError('compiled shared dependency differs from frozen baseline')
    if set(result['sources']) != expected_keys:
        raise ValueError('compiled source closure changed')
    compiler = Path('/opt/homebrew/bin/g++-15').resolve()
    if campaign.sha(compiler) != contract['compiler']['sha256'] or result['command'] != compile_command(output, compiler, name):
        raise ValueError('adapter compiler command changed')
    return result


def compile_adapter(directory, contract, candidate, name):
    directory=Path(directory).resolve()
    output = directory / 'build' / candidate['sha256'] / name
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'build.json'
    if manifest.exists():
        validate_build(campaign.record(manifest), contract, candidate, name,
            campaign.record(Path(__file__).with_name('compact_value_bfm_state_adapter.cpp')))
        return campaign.record(manifest)
    sources = {'candidate': campaign.copy_checked(campaign.verify(candidate), output / 'candidate.cpp')}
    sources['adapter'] = campaign.copy_checked(Path(__file__).with_name('compact_value_bfm_state_adapter.cpp'), output / 'adapter.cpp')
    for filename, item in contract['opponents'][name].items():
        sources['opponent/' + filename] = campaign.copy_checked(campaign.verify(item), output / 'opponent' / filename)
    for path in shared_paths():
        route = str(path.relative_to(campaign.REPO))
        original = subprocess.check_output(['git', 'show', campaign.START + ':' + route], cwd=campaign.REPO)
        if hashlib.sha256(original).hexdigest() != campaign.sha(path):
            raise ValueError('shared referee/search dependency differs from frozen baseline: ' + route)
        sources[route] = campaign.copy_checked(path, output / route)
    compiler = Path('/opt/homebrew/bin/g++-15').resolve()
    if campaign.sha(compiler) != contract['compiler']['sha256']:
        raise ValueError('suite compiler changed')
    binary = output / 'adapter.bin'
    command = compile_command(output, compiler, name)
    with (output / 'build.log').open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, env={**os.environ, **campaign.THREADS})
    campaign.seal(manifest, {'schema': campaign.ID + '.opponent-adapter-build.v2', 'candidate': candidate,
        'opponent': contract['opponents'][name], 'sources': sources, 'compiler': contract['compiler'],
        'command': command, 'binary': campaign.record(binary), 'policy': POLICY})
    return campaign.record(manifest)


def execute_pair(directory, name, row, ordinal, bank, builds):
    output = directory / 'games' / name / row['root_id']
    output.mkdir(parents=True, exist_ok=True)
    arms = {}
    order = ('candidate', 'control') if ordinal % 2 == 0 else ('control', 'candidate')
    for arm in order:
        build = campaign.read(campaign.verify(builds[arm + ':' + name]))
        command = [str(campaign.verify(build['binary'])), str(campaign.verify(bank['tsvs'][name + ':' + row['root_id']])), '800', '155']
        raw = output / (arm + '.jsonl'); stderr = output / (arm + '.stderr')
        started = time.monotonic(); timed_out = False
        with raw.open('xb') as out, stderr.open('xb') as err:
            try:
                result = subprocess.run(command, stdout=out, stderr=err, env={**os.environ, **campaign.THREADS},
                    timeout=POLICY['pair_watchdog_seconds'])
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                timed_out = True; returncode = None
        arms[arm] = {'command': command, 'returncode': returncode, 'timeout': timed_out,
            'elapsed_seconds': time.monotonic() - started, 'output': campaign.record(raw), 'stderr': campaign.record(stderr),
            'build': builds[arm + ':' + name]}
    receipt = output / 'pair.json'
    campaign.seal(receipt, {'schema': campaign.ID + '.opponent-pair-execution.v2', 'opponent': name,
        'root': row, 'arm_order': list(order), 'arms': arms})
    return campaign.record(receipt)


def run(root, context, phase, stage):
    selected,selection_path=source_selection(context,phase)
    directory = context / phase / 'multi-opponent' / stage
    if (directory / 'assessment.json').exists(): return _completed_suite(directory)[0]
    if (directory / 'execution-claim.json').exists():
        raise ValueError('claimed opponent suite is spent; preserve partial evidence, do not restart')
    contract = campaign.read(context / 'campaign.json')
    parent = campaign.read(root / 'campaign.json')
    if campaign.verify(contract['parent_campaign']).parent != root:
        raise ValueError('suite campaign parent changed')
    if not selected.get('eligible_for_multi_opponent'):
        raise ValueError('no eligible full model selected')
    sources = {'candidate': selected['selected']['source'], 'control': parent['inputs']['discrete_v3_deployment.cpp']}
    if sources['candidate']['sha256'] == sources['control']['sha256']:
        raise ValueError('candidate equals deployed control')
    bank = prepare_bank(context, phase, stage, selection_path)
    builds = {arm + ':' + name: compile_adapter(directory, contract, source, name)
        for arm, source in sources.items() for name in campaign.OPPONENTS}
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError('actual-clock evaluation requires nice zero')
    claim_path = directory / 'execution-claim.json'
    campaign.seal(claim_path, {'schema': campaign.ID + '.opponent-suite-execution-claim.v2',
        'bank': campaign.record(directory / 'bank.json'), 'selection': campaign.record(selection_path),
        'sources': sources, 'builds': builds, 'policy': POLICY,
        'driver': campaign.copy_checked(Path(__file__), directory / 'driver.py'),
        'adapter_source': campaign.copy_checked(Path(__file__).with_name('compact_value_bfm_state_adapter.cpp'), directory / 'adapter-source.cpp'),
        'assessment_driver': campaign.copy_checked(Path(evaluation.__file__), directory / 'assessment-driver.py')})
    tasks = [(name, row, i) for name in campaign.OPPONENTS for i, row in enumerate(bank['rows'][name])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(execute_pair, directory, name, row, i, bank, builds) for name, row, i in tasks]
        receipts = [future.result() for future in futures]
    manifest_path = directory / 'manifest.json'
    campaign.seal(manifest_path, {'schema': campaign.ID + '.opponent-suite-execution.v2',
        'claim': campaign.record(claim_path), 'bank': campaign.record(directory / 'bank.json'),
        'builds': builds, 'pairs': receipts})
    arms = {arm: {name: {} for name in campaign.OPPONENTS} for arm in sources}
    incomplete = []
    for record in receipts:
        receipt = campaign.read(campaign.verify(record))
        for arm, execution in receipt['arms'].items():
            if execution['returncode'] != 0 or execution['timeout']:
                incomplete.append({'pair': record, 'arm': arm, 'returncode': execution['returncode'], 'timeout': execution['timeout']})
            else:
                arms[arm][receipt['opponent']].update(checked_games(campaign.verify(execution['output']), receipt['root'], receipt['opponent']))
    if incomplete:
        campaign.seal(directory / 'failure.json', {'schema': campaign.ID + '.opponent-suite-failure.v2',
            'manifest': campaign.record(manifest_path), 'incomplete': incomplete, 'retry_allowed': False})
        raise ValueError('native suite incomplete; partial evidence preserved and claim spent')
    assessment = evaluation.assess(arms['candidate'], arms['control'])
    result = campaign.seal(directory / 'assessment.json', {**assessment, 'manifest': campaign.record(manifest_path)})
    _completed_suite(directory)
    campaign.event(root, 'multi-opponent-' + stage + '-completed', {'assessment': campaign.record(directory / 'assessment.json'),
        'passed': result['passed'], 'campaign_success': False})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--stage', choices=tuple(PAIRS), required=True)
    args = parser.parse_args()
    with campaign.lease(args.root.resolve()):
        result = run(args.root.resolve(), args.context.resolve(), args.phase, args.stage)
    print(json.dumps({'passed': result['passed'], 'improvement': result['equal_weight_improvement']}))


if __name__ == '__main__': main()
