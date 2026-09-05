#!/usr/bin/env python3
"""Attribute the frozen search workload without using diagnostic time for retention.

Run after ordinary uncontended measurements. Every diagnostic execution must
reproduce the corresponding ordinary fixed-work trace. Derivative sources are
recreated by the validator, and category shares and observed timing overhead are
recomputed from native output. Caller holds the campaign's heavy-stage lease.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_search_v2 as search
from tools import compact_value_bfm_timing_instrumentation_v2 as instrumentation

PROBE = campaign.REPO / 'tools/compact_value_bfm_timing_probe.cpp'
SCHEMA = campaign.ID + '.category-profile.v2'
POLICY = {
    'schema': campaign.ID + '.category-profile-policy.v2',
    'profile': 'standard-v1', 'workers': 1, 'threads_per_worker': 1,
    'process_nice': 0, 'mode': 'fixed',
    'schedule': 'all-ordinary-fixed-repetitions-in-frozen-order',
    'interval': 'emergency-action+search+selected-action-encoding',
    'shares': 'exclusive-nanoseconds/total-instrumented-interval-nanoseconds',
    'overhead': 'observed-instrumented/paired-ordinary-fixed-time-minus-one',
    'overhead_interpretation': 'includes-timer-cost-and-between-batch-timing-variation',
    'timing_retention_input': False,
    'required_parity': 'root/action/fixed-trace/nodes/expansions/generated/evaluated',
    'retry_claimed_execution': False, 'qualification_passed': False,
}


def schedule(plan):
    if plan.get('profile') != POLICY['profile'] or set(plan['variants']) != set(search.maintained.SEARCH_VARIANT_ORDER):
        raise ValueError('category profiling requires exactly the four standard search arms')
    return [(repeat, name) for repeat, mode, name in search.measurement_schedule(plan) if mode == 'fixed']


def producer_records():
    return {key: campaign.record(path) for key, path in {
        'driver': Path(__file__), 'instrumentation': Path(instrumentation.__file__),
        'category_probe': PROBE, 'ordinary_probe': search.PROBE,
        'search_validator': Path(search.__file__), 'campaign': Path(campaign.__file__),
    }.items()}


def inputs(plan, output):
    return {'plan': campaign.record(output / 'plan.json'),
            'ordinary_measurement': campaign.record(output / 'measurement.json'),
            'ordinary_claim': campaign.record(output / 'measurement-claim.json'),
            'roots': plan['roots_tsv'], 'runtime': plan['model']['runtime'],
            'compiler': plan['compiler']}


def compile_command(plan, derivative, binary, probe=None):
    return [plan['compiler']['path'], '-std=c++20', '-O3',
            '-DCOMPACT_ENGINE_SOURCE="' + str(derivative) + '"', str(PROBE if probe is None else probe), '-o', str(binary)]


def command(plan, binary):
    return [str(binary), plan['roots_tsv']['path'], 'fixed']


def ordinary_runs(plan, output):
    # This also reproduces the uninstrumented speed gates and cross-arm parity.
    search.validate_measurement(plan, output)
    measurement = campaign.read(output / 'measurement.json')
    return {(row['repeat'], row['variant']): row for row in measurement['runs'] if row['mode'] == 'fixed'}


def validate_raw(raw, ordinary, plan, manifest):
    instrumentation.validate_probe_parity(ordinary, raw, manifest)
    search.validate_probe({**raw, 'schema': 'papersoccer.compact-engine-version-probe.v2'}, plan, 'fixed')
    for row in raw['rows']:
        # Native milliseconds and exclusive timers share one integer ns interval.
        if not math.isclose(row['milliseconds'] * 1_000_000, row['total_search_ns'], rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError('category milliseconds differ from the native instrumented interval')


def summary(plan, native, ordinary):
    result = {}
    for name in plan['variants']:
        pairs = [(native[(repeat, arm)], ordinary[(repeat, arm)])
                 for repeat, arm in schedule(plan) if arm == name]
        attributed_rows = [row for attributed, _ in pairs for row in attributed['rows']]
        ordinary_rows = [row for _, baseline in pairs for row in baseline['rows']]
        keys = list(attributed_rows[0]['category_exclusive_ns'])
        times = {key: sum(row['category_exclusive_ns'][key] for row in attributed_rows) for key in keys}
        calls = {key: sum(row['category_calls'][key] for row in attributed_rows) for key in keys}
        total_ns = sum(row['total_search_ns'] for row in attributed_rows)
        ordinary_ns = sum(row['milliseconds'] for row in ordinary_rows) * 1_000_000
        result[name] = {
            'rows': len(attributed_rows), 'total_instrumented_ns': total_ns,
            'exclusive_ns': times, 'category_calls': calls,
            'shares': {key: value / total_ns for key, value in times.items()},
            'ordinary_fixed_milliseconds': ordinary_ns / 1_000_000,
            'instrumentation_overhead': {
                'observed_elapsed_ratio_minus_one': total_ns / ordinary_ns - 1,
                'observed_added_milliseconds': (total_ns - ordinary_ns) / 1_000_000,
                'interpretation': POLICY['overhead_interpretation'],
            },
            'exact_fixed_work_parity': True, 'categories_reconcile': sum(times.values()) == total_ns,
            'timing_retention_input': False,
        }
    return result


def receipt_body(plan, output, claim, runs, native, ordinary):
    return {'schema': SCHEMA, 'inputs': inputs(plan, output), 'policy': POLICY,
            'claim': campaign.record(claim), 'runs': runs,
            'variants': summary(plan, native, ordinary), 'complete': True,
            'timing_retention_input': False, 'qualification_passed': False, 'campaign_success': False}


def _validate(plan, output):
    path = output / 'category-profile.json'
    receipt = campaign.read(path)
    if (receipt.get('schema') != SCHEMA or receipt.get('complete') is not True
            or receipt.get('timing_retention_input') is not False
            or receipt.get('qualification_passed') is not False or receipt.get('campaign_success') is not False):
        raise ValueError('category profile requires a source-bound native execution receipt')
    expected_schedule = schedule(plan)
    baseline_runs = ordinary_runs(plan, output)
    claim_path = campaign.verify(receipt['claim'])
    claim = campaign.read(claim_path)
    work = output / 'category'
    # Immutable snapshots may relocate unchanged validation code. Preserve the
    # original execution paths, but require every current implementation byte.
    current_producers = producer_records()
    producers_match = (set(claim['producers']) == set(current_producers)
        and all(all(record[field] == current_producers[name][field] for field in ('bytes', 'sha256'))
                for name, record in claim['producers'].items()))
    if (claim_path != work / 'claim.json' or claim.get('schema') != campaign.ID + '.category-profile-claim.v2'
            or claim.get('inputs') != inputs(plan, output) or claim.get('policy') != POLICY
            or not producers_match
            or claim.get('schedule') != [list(row) for row in expected_schedule]
            or claim.get('environment') != campaign.THREADS
            or type(claim.get('workers')) is not int or claim['workers'] != 1
            or type(claim.get('process_nice')) is not int or claim['process_nice'] != 0
            or set(claim['builds']) != set(plan['variants'])
            or [(row['repeat'], row['variant']) for row in receipt['runs']] != expected_schedule
            or any(type(row['repeat']) is not int for row in receipt['runs'])):
        raise ValueError('category profile claim, source closure or sequential schedule changed')
    for record in claim['inputs'].values():
        campaign.verify(record)
    for record in claim['producers'].values():
        campaign.verify(record)
    runtime = claim['host_runtime']
    campaign.verify(runtime['python'])
    if (set(runtime) != {'python', 'python_version', 'platform', 'machine'}
            or any(not isinstance(runtime[key], str) or not runtime[key]
                   for key in ('python_version', 'platform', 'machine'))):
        raise ValueError('category profiling host runtime binding is incomplete')
    manifests = {}
    for name, build in claim['builds'].items():
        source = campaign.verify(plan['variants'][name]['source'])
        derivative_bytes, expected_manifest = instrumentation.instrument_source(source.read_bytes(), campaign.sha(source))
        derivative = campaign.verify(build['instrumented_source'])
        manifest_path = campaign.verify(build['instrumentation_manifest'])
        manifest = campaign.read(manifest_path)
        binary = campaign.verify(build['binary'])
        if (build['original_source'] != plan['variants'][name]['source']
                or derivative != work / 'sources' / (name + '.cpp')
                or derivative.read_bytes() != derivative_bytes
                or manifest_path != work / 'sources' / (name + '.instrumentation.json')
                or {key: value for key, value in manifest.items() if key != 'body_sha256'} != expected_manifest
                or binary != work / 'builds' / name / 'probe.bin'
                or build['command'] != compile_command(plan, derivative, binary,
                    probe=claim['producers']['category_probe']['path'])):
            raise ValueError('category derivative, transformation anchors or compiler binding changed')
        for stream in ('stdout', 'stderr'):
            log = campaign.verify(build['compiler_' + stream])
            if log != binary.parent / ('compiler.' + stream):
                raise ValueError('category compile log belongs to another build')
        manifests[name] = manifest
    native, ordinary = {}, {}
    for run in receipt['runs']:
        key = (run['repeat'], run['variant'])
        repeat, name = key
        baseline = baseline_runs[key]
        if (run['ordinary_run'] != baseline
                or run['command'] != command(plan, claim['builds'][name]['binary']['path'])
                or type(run['returncode']) is not int or run['returncode'] != 0
                or type(run['process_nice']) is not int or run['process_nice'] != 0
                or run['environment'] != campaign.THREADS):
            raise ValueError('category execution changed its original workload or runtime command')
        raw_path = campaign.verify(run['output'])
        stderr = campaign.verify(run['stderr'])
        if (raw_path != work / 'measurements' / f'{repeat}-fixed-{name}.json'
                or stderr != raw_path.with_suffix('.stderr') or stderr.read_bytes()):
            raise ValueError('category output path or native stderr is invalid')
        native[key] = json.loads(raw_path.read_bytes())
        ordinary[key] = json.loads(campaign.verify(baseline['output']).read_bytes())
        validate_raw(native[key], ordinary[key], plan, manifests[name])
    expected = receipt_body(plan, output, claim_path, receipt['runs'], native, ordinary)
    if {key: value for key, value in receipt.items() if key != 'body_sha256'} != expected:
        raise ValueError('category shares or overhead differ from source-bound native evidence')
    return receipt


def validate(plan, output):
    """Validate completed evidence; asserted summaries have no advancement authority."""
    try:
        return _validate(plan, Path(output).resolve())
    except (KeyError, TypeError, IndexError, AttributeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError('malformed source-bound category profile evidence') from error


def produce(root, context, phase):
    """Run the diagnostic batch sequentially while the caller owns the global lease."""
    plan = search.validate_plan(root, context, phase)
    output = search.directory(context, phase)
    if (output / 'category-profile.json').exists():
        return validate(plan, output)
    expected_schedule = schedule(plan)
    baseline_runs = ordinary_runs(plan, output)
    if os.getpriority(os.PRIO_PROCESS, 0) != 0:
        raise ValueError('category profiling requires nice zero')
    work = output / 'category'
    claim_path = work / 'claim.json'
    if claim_path.exists() or list(work.glob('builds/*/probe.bin')) or list(work.glob('measurements/*')):
        raise ValueError('partial category batch requires source-bound review; no automatic retry')
    for record in inputs(plan, output).values():
        campaign.verify(record)
    builds, manifests = {}, {}
    for name in plan['variants']:
        source_record = plan['variants'][name]['source']
        source = campaign.verify(source_record)
        derivative_bytes, manifest = instrumentation.instrument_source(source.read_bytes(), source_record['sha256'])
        derivative = work / 'sources' / (name + '.cpp')
        manifest_path = work / 'sources' / (name + '.instrumentation.json')
        campaign.once(derivative, derivative_bytes)
        campaign.seal(manifest_path, manifest)
        binary = work / 'builds' / name / 'probe.bin'
        binary.parent.mkdir(parents=True, exist_ok=True)
        compile_args = compile_command(plan, derivative, binary)
        compiled = subprocess.run(compile_args, capture_output=True, env={**os.environ, **campaign.THREADS})
        campaign.once(binary.parent / 'compiler.stdout', compiled.stdout)
        campaign.once(binary.parent / 'compiler.stderr', compiled.stderr)
        compiled.check_returncode()
        builds[name] = {'original_source': source_record,
            'instrumented_source': campaign.record(derivative), 'instrumentation_manifest': campaign.record(manifest_path),
            'binary': campaign.record(binary), 'command': compile_args,
            'compiler_stdout': campaign.record(binary.parent / 'compiler.stdout'),
            'compiler_stderr': campaign.record(binary.parent / 'compiler.stderr')}
        manifests[name] = manifest
    campaign.seal(claim_path, {'schema': campaign.ID + '.category-profile-claim.v2',
        'inputs': inputs(plan, output), 'policy': POLICY, 'producers': producer_records(),
        'schedule': [list(row) for row in expected_schedule], 'builds': builds,
        'environment': campaign.THREADS, 'workers': 1, 'process_nice': 0,
        'host_runtime': {'python': campaign.record(Path(sys.executable).resolve()),
                         'python_version': platform.python_version(), 'platform': platform.platform(),
                         'machine': platform.machine()}})
    runs, native, ordinary = [], {}, {}
    for repeat, name in expected_schedule:
        if os.getpriority(os.PRIO_PROCESS, 0) != 0:
            raise ValueError('category execution lost nice zero')
        key = (repeat, name)
        args = command(plan, builds[name]['binary']['path'])
        path = work / 'measurements' / f'{repeat}-fixed-{name}.json'
        finished = subprocess.run(args, capture_output=True, env={**os.environ, **campaign.THREADS})
        campaign.once(path, finished.stdout)
        campaign.once(path.with_suffix('.stderr'), finished.stderr)
        finished.check_returncode()
        if finished.stderr:
            raise ValueError('category profiling emitted unexpected stderr')
        native[key] = json.loads(finished.stdout)
        ordinary[key] = json.loads(campaign.verify(baseline_runs[key]['output']).read_bytes())
        validate_raw(native[key], ordinary[key], plan, manifests[name])
        runs.append({'repeat': repeat, 'variant': name, 'ordinary_run': baseline_runs[key],
                     'command': args, 'output': campaign.record(path), 'stderr': campaign.record(path.with_suffix('.stderr')),
                     'returncode': finished.returncode, 'environment': campaign.THREADS, 'process_nice': 0})
    campaign.seal(output / 'category-profile.json', receipt_body(plan, output, claim_path, runs, native, ordinary))
    return validate(plan, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--phase', required=True)
    parser.add_argument('command', choices=('run', 'validate'))
    args = parser.parse_args()
    with campaign.lease(args.root):
        result = (produce(args.root, args.context, args.phase) if args.command == 'run' else
                  validate(search.validate_plan(args.root, args.context, args.phase), search.directory(args.context, args.phase)))
    print(json.dumps({'schema': result['schema'], 'complete': result['complete'],
                      'timing_retention_input': False, 'qualification_passed': False, 'campaign_success': False}))


if __name__ == '__main__':
    main()
