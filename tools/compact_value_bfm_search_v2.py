#!/usr/bin/env python3
"""Freeze and measure exact full-model search variants before strength selection.

The maintained 2x2 varies feature construction/sorting and descendant selection.
Full/delta inference is an invariant, not another runtime switch. `prepare` and
`measure` do not authorize a source for qualification. `assess` additionally needs
source-bound actual-clock Rank4 executions at search/strength/index.json. That
index must name the shared frozen bank, each exact execution claim and raw result;
no missing measurement or baseline fallback is considered a completed ablation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_full_selection_v2 as full_selection
from tools import compact_value_bfm_teacher_training as maintained

PROBE = campaign.REPO / 'tools/compact_value_bfm_engine_probe.cpp'
GATE_SOURCE = maintained.GATE_SOURCE.with_name('rank4_gate_trajectories.cpp')
POLICY = {
    'schema': campaign.ID + '.search-policy.v2',
    'axes': ['feature-construction/sorting', 'descendant-sort/single-pass-selection'],
    'full_delta': 'invariant-only;all-four-variants-use-delta-inference',
    'default_throughput_profile': 'standard-v1',
    'independent_profiles_only': True, 'cross_turn_persistence': False,
    'minimum_throughput_gain': .10, 'maximum_p95_regression': .05,
    'throughput': 'total-fixed-work-tree-nodes/total-search-seconds',
    'p95': 'nearest-rank-fixed-work-search-milliseconds',
    'clocked_strength': 'maintained-independent-ab-retention',
    'clocked_strength_pairs_per_variant': maintained.FULL_PAIRS,
    'clocked_strength_minimum_candidate_wins': maintained.FULL_MINIMUM_WINS,
    'source_reserve': 2000, 'source_limit_exclusive': 95000,
    'workers': 1, 'threads_per_worker': 1, 'repetitions': 3,
    'profiling_depths': [8, 12, 20, 40], 'profiling_roots_per_depth': 8,
    'profiling_roots': 'first-canonical-full-training-position-per-depth',
    'timing_category_shares': 'source-bound-exclusive-native-timers;diagnostic-time-excluded-from-retention',
    'source_bound_category_profile_required': True,
    'qualification_passed': False, 'campaign_success': False,
}


def directory(context, phase):
    return Path(context).resolve() / phase / 'search'


def full_model(root, context, phase):
    """Reopen the real full selection and its exact six-seed training lineage."""
    contract, _ = full_selection.validate_context(root, context, phase)
    path = Path(context).resolve() / phase / 'full-model-selection.json'
    selected = campaign.read(path)
    training_path = Path(context).resolve() / phase / 'training.json'
    training = campaign.read(training_path)
    rows = full_selection.validate_roster(training, contract['full_training_roster']['lambdas'])
    model = selected.get('selected')
    if (selected.get('schema') != campaign.ID + '.full-model-selection.v2'
            or selected.get('context') != campaign.record(Path(context) / 'campaign.json')
            or selected.get('training') != campaign.record(training_path)
            or selected.get('parent_campaign') != contract['parent_campaign']
            or selected.get('admitted_pilot') != contract['admitted_pilot']
            or selected.get('eligible_for_multi_opponent') is not True or not model
            or model not in selected.get('arms', []) or model.get('lambda') not in (.1, .25)
            or model.get('canonical_retention_passed') is not True
            or model.get('eligible_for_multi_opponent') is not True):
        raise ValueError('search requires an eligible actual full-model selection')
    policy = campaign.read(campaign.verify(selected['policy']))
    expected_policy = full_selection.policy(contract)
    if any(policy.get(key) != value for key, value in expected_policy.items()):
        raise ValueError('full model selection policy changed')
    for item in policy['source_closure']:
        campaign.verify(item)
    for item in selected['seed_references']:
        campaign.verify(item)
    if model['seed_reference'] not in selected['seed_references']:
        raise ValueError('selected full model lost its completed seed reference')
    matches = [row for row in rows if (row['weight'], row['seed']) == (model['lambda'], model['seed'])]
    if len(matches) != 1 or any(matches[0][key] != model[key] for key in ('source', 'runtime', 'float_checkpoint')):
        raise ValueError('search source differs from actual selected training row')
    trainer = full_selection.trainer
    trained = matches[0]
    receipt = trained['seed_receipt']
    seed_directory = Path(context).resolve() / phase / 'training' / f'lambda-{model["lambda"]:.2f}'
    architecture, quantized, _, _ = trainer.load_runtime(campaign.verify(model['runtime']))
    reference = trainer._seed_reference_path(seed_directory, architecture, trainer.ARMS['search-target'], model['seed'])
    gate = trainer.offline_advancement_gate(receipt['float_validation'], receipt['quantized_validation'])
    chosen = maintained._selected_seed([row['seed_receipt'] for row in rows if row['weight'] == model['lambda']])
    if (campaign.record(reference) != model['seed_reference']
            or trainer._load_seed_receipt_from_reference(seed_directory, reference, receipt['binding']) != receipt
            or gate != receipt['offline_gate'] or gate['passed'] is not True or chosen['seed'] != model['seed']):
        raise ValueError('search candidate lost its passing selected full-training receipt')
    initial = trainer.load_float_checkpoint(campaign.verify(contract['inputs']['attempt_one_initial_checkpoint']), architecture)
    parameters = trainer.load_float_checkpoint(campaign.verify(model['float_checkpoint']), architecture)
    full_selection.verify_master_updates(trained, initial, parameters, architecture, quantized)
    full_selection.verify_source_export(matches[0])
    runtime = campaign.read(campaign.verify(model['runtime']))
    if (model['runtime_body_sha256'] != runtime['body_sha256']
            or model['payload_sha256'] != runtime['quantization']['payload_sha256']):
        raise ValueError('selected full model payload changed')
    return contract, selected, path


def profiling_rows(positions):
    """Freeze unprotected existing full-train states, without model feedback."""
    rows = []
    seen = set()
    for depth in POLICY['profiling_depths']:
        candidates = sorted((row for row in positions['rows'] if row['split'] == 'train'
                             and row['drawn_edges'] == depth), key=lambda row: (row['canonical'], row['position_id']))
        kept = []
        for row in candidates:
            if row['canonical'] in seen:
                continue
            state = campaign.features.ReplayState()
            for turn in row['prefix'].split('/'):
                campaign.features.apply_complete_turn(state, state.to_move, turn)
            if state.winner is not None or len(state.used_segments) != depth:
                raise ValueError('profiling position is not a legal nonterminal full-training state')
            if campaign.fingerprints(state)[campaign.legacy.STATE_FINGERPRINT_DOMAIN] != row['canonical']:
                raise ValueError('profiling position canonical identity differs from legal replay')
            seen.add(row['canonical'])
            kept.append({'root_id': f'profile-{depth}-{len(kept):02d}', 'position_id': row['position_id'],
                         'transcript': row['prefix'], 'canonical': row['canonical'], 'edges': depth})
            if len(kept) == POLICY['profiling_roots_per_depth']:
                break
        if len(kept) != POLICY['profiling_roots_per_depth']:
            raise ValueError('full training corpus lacks the frozen profiling depth coverage')
        rows.extend(kept)
    return rows


def prepare(root, context, phase, profile='standard-v1'):
    contract, selected, model_path = full_model(root, context, phase)
    output = directory(context, phase)
    variants = maintained.active_search_variants(profile)
    base = campaign.verify(selected['selected']['source']).read_bytes()
    arms = {}
    for name, macros in variants.items():
        for macro in macros:
            if ('defined(' + macro + ')').encode() not in base:
                raise ValueError('requested profile has no actual source control')
        path = output / 'sources' / (name + '.cpp')
        campaign.once(path, maintained._variant_source(base, macros))
        arms[name] = {'source': campaign.record(path),
                      'metadata': maintained._search_variant_metadata(profile, name)}
    positions_path = Path(context).resolve() / phase / 'positions.json'
    positions = campaign.read(positions_path)
    if positions.get('all_retained_groups_preflighted') is not True:
        raise ValueError('profiling roots lack the validated full corpus')
    roots = profiling_rows(positions)
    roots_path = output / 'profiling-roots.tsv'
    campaign.once(roots_path, ('root_id\ttranscript\n' + ''.join(
        row['root_id'] + '\t' + row['transcript'] + '\n' for row in roots)).encode('ascii'))
    campaign.verify(contract['compiler'])
    compiler = Path('/opt/homebrew/bin/g++-15').resolve()
    if campaign.sha(compiler) != contract['compiler']['sha256']:
        raise ValueError('profiling compiler differs from the frozen campaign compiler')
    return campaign.seal(output / 'plan.json', {
        'schema': campaign.ID + '.search-plan.v2', 'policy': POLICY,
        'context': campaign.record(Path(context) / 'campaign.json'),
        'full_model_selection': campaign.record(model_path),
        'full_model_selection_body_sha256': selected['body_sha256'], 'model': selected['selected'],
        'profile': profile, 'variants': arms, 'positions': campaign.record(positions_path),
        'roots': roots, 'roots_tsv': campaign.record(roots_path), 'compiler': campaign.record(compiler),
        'probe_source': campaign.record(PROBE), 'driver': campaign.record(Path(__file__)),
        'maintained_selector': campaign.record(Path(maintained.__file__)),
        'gate_source': campaign.record(GATE_SOURCE),
        'rank4_source': campaign.record(GATE_SOURCE.parent.parent / 'rank_4/submission.cpp'),
        'gate_validator': campaign.record(Path(maintained.gate_support.__file__)),
        'qualification_passed': False, 'campaign_success': False})


def validate_plan(root, context, phase):
    contract, selected, selection_path = full_model(root, context, phase)
    output = directory(context, phase)
    plan = campaign.read(output / 'plan.json')
    variants = maintained.active_search_variants(plan['profile'])
    if (plan.get('schema') != campaign.ID + '.search-plan.v2' or plan.get('policy') != POLICY
            or plan.get('context') != campaign.record(Path(context) / 'campaign.json')
            or plan.get('full_model_selection') != campaign.record(selection_path)
            or plan.get('full_model_selection_body_sha256') != selected['body_sha256']
            or plan.get('model') != selected['selected'] or set(plan['variants']) != set(variants)
            or plan.get('positions') != campaign.record(Path(context) / phase / 'positions.json')
            or plan['compiler']['sha256'] != contract['compiler']['sha256']):
        raise ValueError('search plan model, policy or corpus changed')
    for key in ('driver', 'maintained_selector', 'probe_source', 'compiler', 'roots_tsv',
                'gate_source', 'rank4_source', 'gate_validator'):
        campaign.verify(plan[key])
    base = campaign.verify(plan['model']['source']).read_bytes()
    for name, macros in variants.items():
        arm = plan['variants'][name]
        if (arm['metadata'] != maintained._search_variant_metadata(plan['profile'], name)
                or campaign.verify(arm['source']).read_bytes() != maintained._variant_source(base, macros)):
            raise ValueError('search variant differs from frozen unchanged trained payload and macros')
    expected_tsv = 'root_id\ttranscript\n' + ''.join(row['root_id'] + '\t' + row['transcript'] + '\n' for row in plan['roots'])
    if campaign.verify(plan['roots_tsv']).read_text() != expected_tsv:
        raise ValueError('profiling root order differs from frozen plan')
    if plan['roots'] != profiling_rows(campaign.read(campaign.verify(plan['positions']))):
        raise ValueError('profiling roots differ from deterministic full-training corpus selection')
    return plan


def compile_command(plan, name, binary):
    return [plan['compiler']['path'], '-std=c++20', '-O3',
            '-DCOMPACT_ENGINE_SOURCE="' + plan['variants'][name]['source']['path'] + '"',
            plan['probe_source']['path'], '-o', str(binary)]


def measurement_schedule(plan):
    names = list(maintained.active_search_variants(plan['profile']))
    return [(repeat, mode, name) for repeat in range(POLICY['repetitions'])
            for mode in ('fixed', 'clock') for name in names[repeat:] + names[:repeat]]


def validate_probe(raw, plan, mode):
    if (raw.get('schema') != 'papersoccer.compact-engine-version-probe.v2'
            or raw.get('mode') != mode or raw.get('payload_sha256') != plan['model']['payload_sha256']
            or any(raw.get(key) is not True for key in ('all_actions_legal', 'all_root_actions_legal',
                'actual_model_full_delta_bit_exact', 'all_root_actions_full_delta_bit_exact'))
            or [row['id'] for row in raw.get('rows', [])] != [row['root_id'] for row in plan['roots']]):
        raise ValueError('native profiling payload, roots or invariants differ')
    for row in raw['rows']:
        if (isinstance(row['milliseconds'], bool) or not isinstance(row['milliseconds'], (int, float))
                or not math.isfinite(row['milliseconds']) or row['milliseconds'] <= 0
                or not row['action'] or set(row['action']) - set('01234567') or not row['fixed_trace']
                or any(isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] < 0
                       for key in ('nodes', 'expansions', 'generated_successors', 'evaluated_successors'))):
            raise ValueError('invalid native profiling row')
    return raw


def measure(root, context, phase):
    """Run one uncontended worker; caller holds the global campaign lease."""
    plan = validate_plan(root, context, phase)
    output = directory(context, phase)
    if (output / 'measurement.json').exists():
        validate_measurement(plan, output)
        return campaign.read(output / 'measurement.json')
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError('uncontended profiling requires nice zero')
    claim_path = output / 'measurement-claim.json'
    if claim_path.exists():
        raise ValueError('partial timing batch requires explicit source-bound review; no automatic timing cherry-pick')
    builds = {}
    for name in maintained.active_search_variants(plan['profile']):
        binary = output / 'builds' / name / 'probe.bin'
        binary.parent.mkdir(parents=True, exist_ok=True)
        command = compile_command(plan, name, binary)
        finished = subprocess.run(command, capture_output=True, check=True)
        campaign.once(binary.parent / 'compiler.stdout', finished.stdout)
        campaign.once(binary.parent / 'compiler.stderr', finished.stderr)
        builds[name] = {'source': plan['variants'][name]['source'], 'binary': campaign.record(binary), 'command': command}
    campaign.seal(claim_path, {'plan': campaign.record(output / 'plan.json'), 'builds': builds,
        'policy': POLICY, 'environment': campaign.THREADS, 'process_nice': 0,
        'schedule': [list(row) for row in measurement_schedule(plan)], 'workers': 1})
    runs = []
    for repeat, mode, name in measurement_schedule(plan):
        path = output / 'measurements' / f'{repeat}-{mode}-{name}.json'
        command = [builds[name]['binary']['path'], plan['roots_tsv']['path'], mode]
        finished = subprocess.run(command, capture_output=True, check=True, env={**os.environ, **campaign.THREADS})
        if finished.stderr:
            raise ValueError('native profiling emitted unexpected stderr')
        validate_probe(json.loads(finished.stdout), plan, mode)
        campaign.once(path, finished.stdout)
        runs.append({'repeat': repeat, 'mode': mode, 'variant': name, 'command': command,
                     'output': campaign.record(path), 'returncode': finished.returncode})
    return campaign.seal(output / 'measurement.json', {'schema': campaign.ID + '.search-measurement.v2',
        'plan': campaign.record(output / 'plan.json'), 'claim': campaign.record(claim_path),
        'runs': runs, 'category_shares': None, 'category_status': POLICY['timing_category_shares']})


def percentile95(values):
    values = sorted(values)
    if not values:
        raise ValueError('empty timing distribution')
    return values[math.ceil(.95 * len(values)) - 1]


def measured_comparison(base, treatment):
    def metrics(rows):
        seconds = sum(row['milliseconds'] for row in rows) / 1000
        nodes = sum(row['nodes'] for row in rows)
        if seconds <= 0 or nodes <= 0:
            raise ValueError('empty measured fixed work')
        return {'nodes_per_second': nodes / seconds, 'p95_ms': percentile95([row['milliseconds'] for row in rows])}
    control, candidate = metrics(base), metrics(treatment)
    throughput = candidate['nodes_per_second'] / control['nodes_per_second'] - 1
    latency = candidate['p95_ms'] / control['p95_ms'] - 1
    return {'control': control, 'candidate': candidate, 'throughput_gain': throughput,
            'p95_regression': latency,
            'passed': candidate['nodes_per_second'] >= (1 + POLICY['minimum_throughput_gain']) * control['nodes_per_second']
            and candidate['p95_ms'] <= (1 + POLICY['maximum_p95_regression']) * control['p95_ms']}


def validate_measurement(plan, output):
    value = campaign.read(output / 'measurement.json')
    claim_path = campaign.verify(value['claim'])
    claim = campaign.read(claim_path)
    expected = measurement_schedule(plan)
    if (value.get('schema') != campaign.ID + '.search-measurement.v2'
            or value.get('plan') != campaign.record(output / 'plan.json')
            or claim_path != output / 'measurement-claim.json' or claim['plan'] != value['plan']
            or claim['policy'] != POLICY or claim['environment'] != campaign.THREADS
            or claim['workers'] != 1 or claim['process_nice'] != 0
            or claim['schedule'] != [list(row) for row in expected]
            or set(claim['builds']) != set(plan['variants'])
            or [(row['repeat'], row['mode'], row['variant']) for row in value['runs']] != expected):
        raise ValueError('measurement claim, roster or one-worker schedule changed')
    rows = {name: {'fixed': [], 'clock': []} for name in plan['variants']}
    for name, build in claim['builds'].items():
        binary = campaign.verify(build['binary'])
        if (build['source'] != plan['variants'][name]['source']
                or build['command'] != compile_command(plan, name, binary)):
            raise ValueError('profiling binary source/compile binding changed')
    for run in value['runs']:
        name, mode = run['variant'], run['mode']
        if run['command'] != [claim['builds'][name]['binary']['path'], plan['roots_tsv']['path'], mode] or run['returncode'] != 0:
            raise ValueError('profiling command or completion changed')
        raw = validate_probe(json.loads(campaign.verify(run['output']).read_bytes()), plan, mode)
        rows[name][mode].extend(raw['rows'])
    # The standard 2x2 is only a speed comparison if every fixed-work trace is exact.
    base_trace = [(row['id'], row['action'], row['fixed_trace']) for row in rows['baseline']['fixed']]
    for name in maintained.SEARCH_VARIANT_ORDER:
        traces = [(row['id'], row['action'], row['fixed_trace']) for row in rows[name]['fixed']]
        if traces != base_trace:
            raise ValueError('standard 2x2 changed fixed-work semantics')
        count = len(plan['roots'])
        if any(traces[start:start + count] != traces[:count] for start in range(0, len(traces), count)):
            raise ValueError('fixed-work repeated trace is nondeterministic')
    comparisons = {}
    for name in plan['variants']:
        if name == 'baseline':
            continue
        base = plan['variants'][name]['metadata']['standard_base_variant'] if '--' in name else 'baseline'
        comparisons[name] = measured_comparison(rows[base]['fixed'], rows[name]['fixed'])
    return comparisons, rows


def validate_strength(plan, output):
    """Consume actual native results, never summaries or asserted pass flags.

    Each execution has claim/raw/returncode. Each claim binds plan, variant,
    source, runtime, bank, binary, compiler, gate_source, rank4_source,
    compile_command, command, workers=1, environment and retry_allowed=False.
    The source gate is the maintained standalone rank4_gate.cpp closure.
    """
    index_path = output / 'strength/index.json'
    index = campaign.read(index_path)
    from tools import compact_value_bfm_search_strength_v2 as strength
    frozen_bank = strength.validate_bank(plan, output / 'strength')
    if (index.get('bank_receipt') != campaign.record(output / 'strength/bank.json')
            or index['bank'] != frozen_bank['tsv']):
        raise ValueError('actual-clock search bank differs from source-bound isolated bank receipt')
    bank = campaign.verify(index['bank'])
    maintained.gate_support.validate_bank(bank)
    if (index.get('schema') != campaign.ID + '.search-strength-index.v2'
            or index['plan'] != campaign.record(output / 'plan.json')
            or set(index['executions']) != set(plan['variants'])):
        raise ValueError('actual-clock search strength roster/plan changed')
    documents, requests = {}, {}
    shared_configuration = None
    for name, record in index['executions'].items():
        destination = output / 'strength' / name
        execution_path = campaign.verify(record)
        execution = campaign.read(execution_path)
        claim_path = campaign.verify(execution['claim'])
        claim = campaign.read(claim_path)
        arm = plan['variants'][name]
        profile = arm['metadata']['candidate_search_profile']
        if (execution_path != destination / 'execution.json' or claim_path != destination / 'claim.json'
                or execution.get('schema') != campaign.ID + '.search-strength-execution.v2'
                or claim.get('schema') != campaign.ID + '.search-strength-claim.v2'
                or claim['plan'] != index['plan'] or claim['variant'] != name or claim['bank'] != index['bank']
                or claim.get('bank_receipt') != index['bank_receipt'] or claim.get('policy') != strength.POLICY
                or type(claim.get('process_nice')) is not int or claim['process_nice'] != 0
                or claim['source'] != arm['source'] or claim['runtime'] != plan['model']['runtime']
                or claim['compiler'] != plan['compiler'] or claim['workers'] != 1
                or claim['environment'] != campaign.THREADS or claim['retry_allowed'] is not False
                or type(execution['returncode']) is not int or execution['returncode'] not in (0, 2)):
            raise ValueError('actual-clock execution source, worker or plan binding changed')
        binary = campaign.verify(claim['binary'])
        gate = campaign.verify(claim['gate_source'])
        rank4 = campaign.verify(claim['rank4_source'])
        if (binary != destination / 'gate.bin'
                or claim['gate_source'] != plan['gate_source'] or claim['rank4_source'] != plan['rank4_source']
                or campaign.sha(rank4) != maintained.gate_support.RANK4_SHA256
                or campaign.sha(gate.parent.parent / 'rank_4/submission.cpp') != campaign.sha(rank4)
                or claim['compile_command'] != [plan['compiler']['path'], '-std=c++20', '-O3',
                    '-DCOMPACT_VALUE_BFM_CANDIDATE_SOURCE="' + arm['source']['path'] + '"',
                    str(gate), '-o', str(binary)]):
            raise ValueError('actual-clock strength compile/source closure changed')
        raw_path = campaign.verify(execution['raw'])
        if raw_path != destination / 'result.json':
            raise ValueError('actual-clock raw result belongs to another variant execution')
        document = maintained.gate_support.validate_result(raw_path, expected_bank_sha256=index['bank']['sha256'],
            expected_candidate_sha256=arm['source']['sha256'], expected_candidate_search_profile=profile,
            require_trajectories=True, trajectory_bank=bank)
        config = document['config']
        if (config['mode'] != 'actual-clock' or config['pair_offset'] != 0
                or config['pair_count'] != POLICY['clocked_strength_pairs_per_variant']
                or config['minimum_candidate_wins'] != POLICY['clocked_strength_minimum_candidate_wins']
                or any(config.get(key) != value for key, value in maintained.GATE_CONFIGURATION.items())
                or execution['returncode'] != (0 if document['result']['passed'] else 2)
                or document['bindings']['candidate_runtime_body_sha256'] != plan['model']['runtime_body_sha256']
                or document['bindings']['candidate_payload_sha256'] != plan['model']['payload_sha256']):
            raise ValueError('actual-clock search model/runtime binding changed')
        normalized = {key: value for key, value in config.items() if key != 'candidate_search_profile'}
        if shared_configuration is not None and normalized != shared_configuration:
            raise ValueError('search variants used different actual-clock configurations')
        shared_configuration = normalized
        command = [str(binary), '--bank', str(bank), '--candidate-source', arm['source']['path'],
            '--rank4-source', str(rank4), '--output', str(raw_path), '--expected-bank-sha256', index['bank']['sha256'],
            '--expected-candidate-sha256', arm['source']['sha256'], '--pair-offset', '0', '--pair-count', str(config['pair_count']),
            '--mode', 'actual-clock', '--minimum-candidate-wins', str(config['minimum_candidate_wins']),
            '--candidate-seed', '1', '--include-trajectories']
        if claim['command'] != command:
            raise ValueError('actual-clock strength command differs from executed evidence')
        documents[name] = document
        requests[name] = {
            'search_throughput_profile': plan['profile'], 'search_variant': name, 'search_variant_metadata': arm['metadata'],
            'compile_time_macros': arm['metadata']['compile_time_macros'], 'macros_embedded_at_source_start': True,
            'source_is_default_for_variant': True, 'source_reserve': 95000 - arm['source']['bytes'],
            'candidate_source': arm['source'], 'binary': claim['binary'], 'configuration': config, 'bank': index['bank'],
            'training_selection': plan['full_model_selection'], 'training_selection_body_sha256': plan['full_model_selection_body_sha256'],
            'ranking_weight': plan['model']['lambda'], 'seed': plan['model']['seed'], 'runtime': plan['model']['runtime'],
            'runtime_body_sha256': plan['model']['runtime_body_sha256'], 'runtime_payload_sha256': plan['model']['payload_sha256'],
            'base_model_source': plan['model']['source'],
        }
    return documents, requests


def category_profile_status(plan, output):
    """Only reconstructed native attribution evidence completes profiling."""
    path = output / 'category-profile.json'
    if not path.exists():
        return {'complete': False, 'receipt': None,
                'reason': 'source-bound native category timing shares have not been produced or validated'}
    from tools import compact_value_bfm_category_profile_v2 as categories
    value = categories.validate(plan, output)
    return {'complete': True, 'receipt': campaign.record(path),
            'variants': value['variants'], 'timing_retention_input': False,
            'reason': 'exclusive native category times reconcile and reproduce the ordinary fixed-work traces'}


def selection_body(root, context, phase):
    plan = validate_plan(root, context, phase)
    output = directory(context, phase)
    comparisons, _ = validate_measurement(plan, output)
    documents, requests = validate_strength(plan, output)
    if plan['profile'] != 'standard-v1':
        # Existing widening/reuse tests prove profile-specific invariants, not
        # exact tree-trace equality. Their source-bound execution bridge must be
        # added before a treatment is eligible; counters alone are insufficient.
        raise ValueError('independent treatment requires source-bound profile invariant execution; selection remains closed')
    summaries = {name: maintained._result_summary(raw) for name, raw in documents.items()}
    cleanliness = maintained._standard_variant_cleanliness(documents, requests, profile=plan['profile'])
    retention = maintained._select_complete_search_variant(summaries, variant_cleanliness=cleanliness)
    retained = []
    for name in retention['retained_variants']:
        if not maintained._zero_failures(summaries[name]) or not maintained._actual_clock_timing_clean(documents[name]):
            continue
        if name != 'baseline' and not comparisons[name]['passed']:
            continue
        if name == 'combined' and any(not comparisons[arm]['passed'] for arm in ('no-feature-sort-only', 'single-pass-selection-only')):
            continue
        retained.append(name)
    preference = {name: index for index, name in enumerate(('combined', 'no-feature-sort-only', 'single-pass-selection-only', 'baseline'))}
    chosen = min(retained, key=lambda name: (-summaries[name]['candidate_wins'], preference[name])) if retained else None
    evaluated = ({**plan['model'], 'source': plan['variants'][chosen]['source'], 'search_variant': chosen,
                 'source_reserve': 95000 - plan['variants'][chosen]['source']['bytes'],
                 'candidate_search_profile': plan['variants'][chosen]['metadata']['candidate_search_profile'],
                 'compile_time_macros': plan['variants'][chosen]['metadata']['compile_time_macros']} if chosen else None)
    category_profile = category_profile_status(plan, output)
    selected = evaluated if category_profile['complete'] else None
    return {'schema': campaign.ID + '.search-selection.v2', 'plan': campaign.record(output / 'plan.json'),
        'full_model_selection': plan['full_model_selection'], 'measurement': campaign.record(output / 'measurement.json'),
        'strength': campaign.record(output / 'strength/index.json'), 'selected': selected,
        'evaluated_candidate': evaluated,
        'retained_variants': retained, 'throughput_and_latency': comparisons, 'clocked_strength': retention,
        'eligible_for_multi_opponent': selected is not None, 'required_ablation_complete': True,
        'required_category_profile_complete': category_profile['complete'],
        'required_profiling_complete': category_profile['complete'], 'category_profile': category_profile,
        'incomplete_requirements': [] if category_profile['complete'] else ['source-bound-native-category-timing-profile'],
        'status': 'awaiting-source-bound-category-profile' if not category_profile['complete'] else
                  'source-selected-awaiting-multi-opponent' if selected else 'search-strength-rejected',
        'category_status': POLICY['timing_category_shares'],
        'qualification_passed': False, 'campaign_success': False}


def assess(root, context, phase):
    body = selection_body(root, context, phase)
    name = 'search-selection.json' if body['required_profiling_complete'] else 'search-assessment-incomplete.json'
    return campaign.seal(directory(context, phase) / name, body)


def validate_selection(root, context, phase):
    """Reproduce the exact result for the downstream source resolver."""
    path = directory(context, phase) / 'search-selection.json'
    actual = campaign.read(path)
    if actual.get('required_profiling_complete') is not True:
        raise ValueError('final search selection lacks completed source-bound category profiling')
    body = selection_body(root, context, phase)
    if {key: value for key, value in actual.items() if key != 'body_sha256'} != body:
        raise ValueError('search selection differs from actual source-bound measurements and games')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('--profile', choices=maintained.SEARCH_THROUGHPUT_PROFILE_ORDER, default='standard-v1')
    parser.add_argument('command', choices=('prepare', 'measure', 'category-profile', 'assess', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root):
        from tools import compact_value_bfm_category_profile_v2 as categories
        command = {'prepare': prepare, 'measure': measure, 'category-profile': categories.produce,
                   'assess': assess, 'validate': validate_selection}[args.command]
        if args.command == 'prepare':
            result = command(args.root, args.context, args.phase, args.profile)
        else:
            result = command(args.root, args.context, args.phase)
    print(json.dumps({'schema': result['schema'], 'eligible_for_multi_opponent': result.get('eligible_for_multi_opponent', False),
                      'qualification_passed': False, 'campaign_success': False}))


if __name__ == '__main__':
    main()
