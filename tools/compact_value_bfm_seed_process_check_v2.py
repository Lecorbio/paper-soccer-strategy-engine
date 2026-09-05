#!/usr/bin/env python3
"""Prepare or run an isolated real-smoke thread/spawn equivalence experiment.

Preparation runs no training and changes no campaign phase. Execution requires
the campaign heavy-stage lease, runs the two rosters sequentially, and never
qualifies a model or enables the production executor automatically.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

if __name__ == '__main__':
    for key in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign
from tools import compact_value_bfm_seed_process_v2 as process
from tools import compact_value_bfm_train as trainer
from tools import compact_value_bfm_teacher_training as adapter
from tools import compact_value_bfm_ranking_store as storage

SCHEMA = campaign.ID + '.seed-process-check.v2'
WEIGHTS = (0.0, .1, .25)
SEEDS = trainer.FIXED_SEEDS[:2]
STAGES = ('threads', 'spawn')


def sources():
    exporter = adapter.source_exporter
    files = {Path(item['path']) for item in process.source_closure()}
    files.update((Path(__file__).resolve(), Path(adapter.model_exporter.__file__),
                  Path(exporter.__file__), exporter.CONFIG))
    config = json.loads(exporter.CONFIG.read_text())
    manifest = exporter.contained(exporter.HERE, config.get('sources', 'sources.txt'), 'sources')
    files.add(manifest)
    files.update(exporter.contained(exporter.ROOT, line.strip(), 'source')
        for line in manifest.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'))
    return [campaign.record(path) for path in sorted(files)]


def output_path(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    allowed = root / 'diagnostics/seed-process-equivalence'
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('executor check output must be a fresh child of diagnostics/seed-process-equivalence')
    return output


def smoke_inputs_metadata(root):
    """Verify historical routing without allocating the million-row anchor."""
    root = Path(root).resolve()
    directory = root / 'smoke-064'
    plan = campaign.read(root / 'campaign.json')
    completion = campaign.read(directory / 'smoke-completion-corrected.json')
    training_path = campaign.verify(completion['training_receipt'])
    if (training_path != directory / 'training.json' or completion['qualification_eligible'] is not False
            or completion['campaign_success'] is not False or completion['new_training_verified'] is not True
            or completion['games'] != 64):
        raise ValueError('a completed real 64-game smoke is required')
    training = campaign.read(training_path)
    audit_path = campaign.verify(training['input_audit'])
    audit = campaign.read(audit_path)
    labels_path = campaign.verify(audit['labels'])
    labels = campaign.read(labels_path)
    # Historical smoke labels removed one duplicate group. The audit binds the
    # full preflight closure; original training filtered anchors from eligible.
    if (audit_path != directory / 'training-input-audit.json'
            or campaign.verify(audit['position_closure']) != directory / 'positions.json'
            or labels_path != directory / 'labels.json'
            or campaign.verify(labels['positions']) != directory / 'eligible-positions.json'
            or audit['bundle'] != plan['bundle'] or audit['protected_tests_opened'] is not False
            or audit['anchor_duplicates_removed'] != 2 or labels['groups'] != 994
            or training.get('smoke') is not True or training.get('mandatory_training_verified') is not True):
        raise ValueError('historical smoke training/eligible-position identity changed')
    store_path = campaign.verify(audit['ranking_store'])
    store = campaign.read(store_path)
    if store['sources'] != [labels['merged']] or len(store['groups']) != 994:
        raise ValueError('smoke ranking store differs from final retained labels')
    for item in (audit['bundle'], audit['exclusion_index'], labels['merged'], *store['arrays'].values()):
        campaign.verify(item)
    for shard in audit['shards'].values():
        for item in shard.values():
            campaign.verify(item)
    roster = {(row['weight'], row['seed']) for row in training['results']}
    if len(training['results']) != 3 or roster != {(weight, SEEDS[0]) for weight in WEIGHTS}:
        raise ValueError('historical smoke must bind exactly its three original seed recipes')
    base = next(row['seed_receipt']['binding'] for row in training['results'] if row['weight'] == 0)
    trainer.verify_body_hash(base, schema='papersoccer.compact-value-bfm-training-binding.v1', label='smoke binding')
    for row in training['results']:
        if row['seed_receipt']['binding'] != process.roster_binding(base, row['seed'], row['weight']):
            raise ValueError('historical smoke training input bindings differ across recipes')
    if (base['settings']['qat_profile'] != trainer.STANDARD_QAT_PROFILE
            or base['input_audit'] != audit
            or base['successor_ranking']['initial_checkpoint']['sha256'] != plan['inputs']['attempt_one_initial_checkpoint']['sha256']):
        raise ValueError('smoke initialization or standard recipe changed')
    return {'campaign': campaign.record(root / 'campaign.json'),
        'completion': campaign.record(directory / 'smoke-completion-corrected.json'),
        'training': campaign.record(training_path), 'audit': campaign.record(audit_path),
        'labels': audit['labels'], 'audited_position_closure': audit['position_closure'],
        'eligible_positions_used_for_anchor_filter': labels['positions'],
        'initial_checkpoint': plan['inputs']['attempt_one_initial_checkpoint'],
        'expected_base_binding': base, 'expected_datasets': base['datasets'],
        'expected_anchor_rows_removed': 2, 'retained_groups': 994}


def prepare(root, output):
    root = Path(root).resolve()
    output = output_path(root, output)
    metadata = smoke_inputs_metadata(root)
    body = {'schema': SCHEMA, 'root': str(root), 'output': str(output),
        'smoke': metadata, 'sources': sources(),
        'python': {'executable': str(Path(sys.executable).resolve()), 'version': sys.version},
        'weights': list(WEIGHTS), 'seeds': list(SEEDS), 'stage_order': list(STAGES),
        'maximum_active_real_seeds': 2, 'numerical_threads_per_seed': 1,
        'spawn_start_method': 'spawn', 'full_maintained_seed_and_qat_execution': True,
        'historical_receipts_reused_as_execution_results': False,
        'real_core_inputs_reconstructed_during_prepare': False,
        'global_heavy_stage_lease_required': True, 'qualification_eligible': False,
        'automatic_production_opt_in': False, 'real_training_started': False,
        'performance_interpretation': 'one ordered paired experiment; report cache/order effects; no automatic speed claim'}
    campaign.seal(output / 'plan.json', body)
    return campaign.record(output / 'plan.json')


def validate_plan(path):
    path = Path(path).resolve()
    plan = campaign.read(path)
    if (plan['schema'] != SCHEMA or output_path(plan['root'], plan['output']) / 'plan.json' != path
            or plan['weights'] != list(WEIGHTS) or plan['seeds'] != list(SEEDS)
            or plan['stage_order'] != list(STAGES) or plan['maximum_active_real_seeds'] != 2
            or plan['numerical_threads_per_seed'] != 1 or plan['spawn_start_method'] != 'spawn'
            or plan['qualification_eligible'] is not False or plan['automatic_production_opt_in'] is not False
            or plan['real_training_started'] is not False
            or plan['python'] != {'executable': str(Path(sys.executable).resolve()), 'version': sys.version}
            or plan['sources'] != sources()):
        raise ValueError('executor check plan/source/runtime changed')
    if plan['smoke'] != smoke_inputs_metadata(plan['root']):
        raise ValueError('executor check historical smoke binding changed')
    return plan


def reconstruct_smoke(plan):
    """Rebuild the original TrainingInputs without redirecting its audit paths."""
    metadata = plan['smoke']
    audit = campaign.read(campaign.verify(metadata['audit']))
    labels = campaign.read(campaign.verify(metadata['labels']))
    campaign.verify(metadata['audited_position_closure'])
    positions = campaign.read(campaign.verify(metadata['eligible_positions_used_for_anchor_filter']))
    if labels['positions'] != metadata['eligible_positions_used_for_anchor_filter']:
        raise ValueError('smoke eligible-position reconstruction changed')
    bundle = trainer.FrozenBundle.load(campaign.verify(audit['bundle']))
    rankings = storage.RankingStore(campaign.verify(audit['ranking_store']), bundle).labels()
    shard = audit['shards']['train']
    manifest, npz = campaign.verify(shard['manifest']), campaign.verify(shard['npz'])
    new = trainer.load_shard(adapter._ExternalShardView(manifest, npz), manifest.name)
    anchor, common, canonical, routes = adapter._load_core_inputs(bundle)
    anchor, filtered = process.filter_early_anchor(anchor, positions['rows'])
    if filtered['removed_rows'] != metadata['expected_anchor_rows_removed']:
        raise ValueError('smoke anchor reconstruction changed')
    inputs = trainer.TrainingInputs(new=new, anchor=anchor, common_adjudicator=common,
        canonical_validation=canonical, source_routes={**routes, 'new': (shard['manifest']['path'],)},
        paired_row_validation={'external_source_bound': True},
        split_isolation={'closure_audit': audit['body_sha256']}, input_audit=audit,
        successor_rankings=rankings)
    observed = trainer.training_binding(bundle, inputs, trainer.ARCHITECTURES['capacity-12x8'],
        trainer.ARMS['search-target'], SEEDS[0], None, 0,
        campaign.verify(metadata['initial_checkpoint']), trainer.STANDARD_QAT_PROFILE)
    if observed != metadata['expected_base_binding']:
        raise ValueError('reconstructed real-smoke inputs differ from historical full training binding')
    identity = process.input_identity(inputs, filtered)
    for dataset in (new, anchor, common, canonical):
        for name in ('indptr', 'indices', 'targets', 'weights', 'group_ids'):
            getattr(dataset, name).flags.writeable = False
    return bundle, inputs, identity


def expected_worker_spec(plan_path, stage, identity):
    plan = campaign.read(plan_path)
    if stage not in STAGES:
        raise ValueError('unknown executor check stage')
    directory = Path(plan['output']) / stage
    jobs = [{'weight': weight, 'seed': seed, 'directory': str(directory / f'lambda-{weight:.2f}'),
        'binding': process.roster_binding(plan['smoke']['expected_base_binding'], seed, weight)}
        for weight in WEIGHTS for seed in SEEDS]
    return {'schema': SCHEMA + '.worker', 'plan': campaign.record(plan_path),
        'stage': stage, 'ranking_weights': list(WEIGHTS), 'seeds': list(SEEDS),
        'jobs': jobs, 'qat_profile': trainer.STANDARD_QAT_PROFILE, 'reconstruction': identity,
        'historical_position_adapter_only': True, 'production_seed_execution_path': True,
        'qualification_eligible': False}


def execution_spec(plan_path, stage, identity):
    plan = campaign.read(plan_path)
    path = Path(plan['output']) / stage / 'worker-spec.json'
    campaign.seal(path, expected_worker_spec(plan_path, stage, identity))
    return path


def expected_claim(plan_path, stage):
    return {'plan': campaign.record(plan_path), 'stage': stage,
        'starts_real_smoke_training': True, 'maximum_active_seeds': 2,
        'qualification_eligible': False, 'retry_allowed': False}


def _initialize_smoke(spec_path):
    spec = campaign.read(spec_path)
    plan = validate_plan(campaign.verify(spec['plan']))
    if Path(spec_path).resolve() != Path(plan['output']) / 'spawn/worker-spec.json':
        raise ValueError('spawn smoke specification is outside its canonical output')
    if spec['schema'] != SCHEMA + '.worker' or spec['stage'] != 'spawn':
        raise ValueError('spawn smoke worker specification changed')
    with trainer.native_thread_execution_scope():
        bundle, inputs, identity = reconstruct_smoke(plan)
    if identity != spec['reconstruction']:
        raise ValueError('spawn smoke input reconstruction differs from thread reference')
    if {key: value for key, value in spec.items() if key != 'body_sha256'} != expected_worker_spec(
            campaign.verify(spec['plan']), 'spawn', identity):
        raise ValueError('spawn smoke job/binding roster changed')
    process._WORKER = (spec, bundle, inputs, campaign.verify(plan['smoke']['initial_checkpoint']))


class SmokeSpawnExecutor(process.SpawnSeedExecutor):
    """Exercise production dispatch/result/shutdown with historical smoke inputs."""
    def __enter__(self):
        os.environ.update(process.ENVIRONMENT)
        os.environ[process.MARKER] = '1'
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=2,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize_smoke,
            initargs=(str(self.path),))
        return self


class MemorySampler:
    """Sample coordinator and descendant RSS; save aggregate and per-PID peaks."""
    def __init__(self):
        self.stop = threading.Event()
        self.peak = 0
        self.pids = {}
        self.samples = 0
        self.errors = []

    def sample(self):
        rows = {}
        for line in subprocess.check_output(['ps', '-axo', 'pid=,ppid=,rss='], text=True).splitlines():
            pid, parent, rss = map(int, line.split())
            rows[pid] = (parent, rss)
        selected = {os.getpid()}
        while True:
            added = {pid for pid, (parent, _rss) in rows.items() if parent in selected} - selected
            if not added:
                break
            selected.update(added)
        values = {pid: rows[pid][1] for pid in selected if pid in rows}
        self.peak = max(self.peak, sum(values.values()))
        for pid, rss in values.items():
            self.pids[pid] = max(self.pids.get(pid, 0), rss)
        self.samples += 1

    def loop(self):
        while not self.stop.is_set():
            try:
                self.sample()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.errors.append(str(error))
            self.stop.wait(2)

    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_error):
        self.stop.set()
        self.thread.join()

    def report(self):
        return {'samples': self.samples, 'interval_seconds': 2, 'units': 'KiB',
            'process_tree_peak_rss': self.peak,
            'per_pid_peak_rss': {str(pid): rss for pid, rss in sorted(self.pids.items())},
            'includes_coordinator_and_descendants': True, 'errors': self.errors}


def usage():
    values = {name: {key: getattr(resource.getrusage(kind), key) for key in
        ('ru_utime', 'ru_stime', 'ru_maxrss', 'ru_minflt', 'ru_majflt')}
        for name, kind in (('coordinator', resource.RUSAGE_SELF), ('reaped_children', resource.RUSAGE_CHILDREN))}
    return {'values': values, 'ru_maxrss_units': 'bytes' if sys.platform == 'darwin' else 'KiB',
        'ru_maxrss_is_lifetime_peak_not_stage_delta': True}


def export_result(job, receipt):
    directory = Path(job['directory'])
    reference = trainer._seed_reference_path(directory, trainer.ARCHITECTURES['capacity-12x8'],
        trainer.ARMS['search-target'], job['seed'])
    if trainer._load_seed_receipt_from_reference(directory, reference, job['binding']) != receipt:
        raise ValueError('executor check receipt does not reproduce its seed reference')
    artifacts = {}
    for key in ('float_checkpoint', 'quantized_runtime'):
        artifact = trainer._output_artifact(directory, receipt[key]['path'],
            expected_sha256=receipt[key]['sha256'], label=key)
        artifacts[key] = campaign.record(artifact)
    source = directory / f'seed-{job["seed"]}.cpp'
    campaign.once(source, adapter._runtime_source(Path(artifacts['quantized_runtime']['path'])))
    source.read_bytes().decode('ascii')
    return {'weight': job['weight'], 'seed': job['seed'], 'reference': campaign.record(reference),
        'receipt': receipt, 'source': campaign.record(source), **artifacts}


def run_stage(plan_path, stage):
    if stage not in STAGES:
        raise ValueError('unknown executor check stage')
    plan = validate_plan(plan_path)
    directory = Path(plan['output']) / stage
    result_path = directory / 'result.json'
    if result_path.exists():
        return validated_stage(plan_path, stage)
    if (directory / 'claim.json').exists():
        raise ValueError('interrupted executor timing is not resumable; preserve it and prepare a new diagnostic run')
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('executor timing requires fresh output directories without preexisting seed artifacts')
    campaign.seal(directory / 'claim.json', expected_claim(plan_path, stage))
    started = time.monotonic()
    before = usage()
    results, process_evidence = [], []
    with MemorySampler() as memory:
        with trainer.native_thread_execution_scope():
            bundle, inputs, identity = reconstruct_smoke(plan)
        spec_path = execution_spec(plan_path, stage, identity)
        spec = campaign.read(spec_path)
        if stage == 'threads':
            initial = campaign.verify(plan['smoke']['initial_checkpoint'])
            with trainer.native_thread_execution_scope() as execution:
                def run_job(job):
                    return trainer.train_seed_candidate(bundle, inputs, trainer.ARCHITECTURES['capacity-12x8'],
                        trainer.ARMS['search-target'], job['seed'], Path(job['directory']),
                        ranking_weight=job['weight'], initial_checkpoint=initial,
                        qat_profile=trainer.STANDARD_QAT_PROFILE, resume=False,
                        _native_thread_execution=execution)
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    for weight in WEIGHTS:
                        jobs = [job for job in spec['jobs'] if job['weight'] == weight]
                        receipts = list(pool.map(run_job, jobs))
                        results.extend(export_result(job, receipt) for job, receipt in zip(jobs, receipts, strict=True))
        else:
            # Match the production coordinator, which retains its reconstructed
            # inputs while the two children have their own ordinary arrays.
            with SmokeSpawnExecutor(spec_path) as executor:
                for weight in WEIGHTS:
                    receipts = executor.run_weight(weight)
                    jobs = [job for job in spec['jobs'] if job['weight'] == weight]
                    results.extend(export_result(job, receipt) for job, receipt in zip(jobs, receipts, strict=True))
                process_evidence = executor.evidence
    return campaign.seal(result_path, {'schema': SCHEMA + '.stage', 'stage': stage,
        'plan': campaign.record(plan_path), 'specification': campaign.record(spec_path),
        'claim': campaign.record(directory / 'claim.json'),
        'reconstruction': identity, 'results': results, 'elapsed_seconds': time.monotonic() - started,
        'usage_before': before, 'usage_after': usage(), 'memory': memory.report(),
        'process_evidence': process_evidence, 'qualification_eligible': False})


def compare_stage_results(threaded, spawned):
    if (threaded['stage'], spawned['stage']) != STAGES or threaded['plan'] != spawned['plan']:
        raise ValueError('executor results belong to different experiments or stages')
    if threaded['reconstruction'] != spawned['reconstruction']:
        raise ValueError('thread/spawn real-smoke reconstructed inputs differ')
    expected = [(weight, seed) for weight in WEIGHTS for seed in SEEDS]
    if any([(row['weight'], row['seed']) for row in stage['results']] != expected for stage in (threaded, spawned)):
        raise ValueError('executor comparison has an incomplete seed roster')
    for stage in (threaded, spawned):
        elapsed = stage['elapsed_seconds']
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('executor elapsed time must be finite and positive')
    checks = []
    for left, right in zip(threaded['results'], spawned['results'], strict=True):
        # The deterministic receipts include all scale trials, selected scales, QAT
        # trajectories, float/quantized metrics, gradients' update evidence,
        # parity checks, and exact training bindings. PID/timing is separate.
        for row in (left, right):
            trainer.validate_native_thread_execution(row['receipt']['native_thread_execution'])
        def deterministic(receipt):
            return {key: value for key, value in receipt.items()
                if key not in ('native_thread_execution', 'body_sha256')}
        equal = deterministic(left['receipt']) == deterministic(right['receipt'])
        artifacts = {key: (left[key]['sha256'], left[key]['bytes']) == (right[key]['sha256'], right[key]['bytes'])
            for key in ('float_checkpoint', 'quantized_runtime', 'source')}
        if not equal or not all(artifacts.values()):
            raise ValueError(f'full maintained seed differs for lambda={left["weight"]}, seed={left["seed"]}')
        checks.append({'weight': left['weight'], 'seed': left['seed'],
            'all_deterministic_receipt_fields_equal': equal, 'artifact_bytes_equal': artifacts,
            'native_execution_contracts_validated': True,
            'native_execution_records_equal': left['receipt']['native_thread_execution'] == right['receipt']['native_thread_execution']})
    workers = sorted({row['process']['pid'] for row in spawned['process_evidence']})
    complete_memory = all(row['memory']['samples'] > 0 and not row['memory']['errors']
        and row['memory'].get('process_tree_peak_rss', 0) > 0
        and bool(row['memory'].get('per_pid_peak_rss')) for row in (threaded, spawned))
    complete_memory = complete_memory and all(str(pid) in spawned['memory']['per_pid_peak_rss'] for pid in workers)
    return {'exact_equivalence_passed': True, 'checks': checks,
        'two_spawn_children_observed': len(workers) == 2, 'spawn_worker_pids': workers,
        'thread_seconds': threaded['elapsed_seconds'], 'spawn_seconds': spawned['elapsed_seconds'],
        'observed_thread_over_spawn_elapsed_ratio': threaded['elapsed_seconds'] / spawned['elapsed_seconds'],
        'memory_measurements_complete': complete_memory,
        'executor_ready_for_review': len(workers) == 2 and complete_memory,
        'performance_is_single_ordered_experiment': True, 'qualification_eligible': False,
        'automatic_production_opt_in': False}


def bound(record, path):
    path = Path(path).absolute()
    if path.resolve() != path or Path(record['path']).absolute() != path:
        raise ValueError('executor check artifact redirected outside its canonical output path')
    campaign.verify(record)
    return path


def validate_reconstruction(identity, plan):
    body = {key: value for key, value in identity.items() if key != 'body_sha256'}
    base = plan['smoke']['expected_base_binding']
    if (identity.get('body_sha256') != hashlib.sha256(campaign.raw(body)).hexdigest()
            or any(identity.get(key) != base[key] for key in
                ('datasets', 'source_routes', 'paired_row_validation', 'split_isolation', 'input_audit'))
            or identity['anchor_filter']['removed_rows'] != plan['smoke']['expected_anchor_rows_removed']
            or identity['anchor_filter']['original_rows'] != base['datasets']['anchor']['samples'] + plan['smoke']['expected_anchor_rows_removed']
            or any(identity['ranking'][key] != base['successor_ranking'][key] for key in
                ('artifact_sha256', 'body_sha256', 'schema', 'source_bundle_body_sha256', 'teacher'))):
        raise ValueError('executor reconstructed identity differs from historical smoke')


def validate_memory(value):
    if (value.get('units') != 'KiB' or value.get('interval_seconds') != 2
            or value.get('includes_coordinator_and_descendants') is not True
            or type(value.get('samples')) is not int or value['samples'] < 0
            or type(value.get('process_tree_peak_rss')) is not int or value['process_tree_peak_rss'] < 0
            or not isinstance(value.get('errors'), list)
            or not isinstance(value.get('per_pid_peak_rss'), dict)):
        raise ValueError('executor memory observation is malformed')
    for pid, rss in value['per_pid_peak_rss'].items():
        if not pid.isdigit() or int(pid) <= 0 or type(rss) is not int or not 0 <= rss <= value['process_tree_peak_rss']:
            raise ValueError('executor memory PID/RSS observation is malformed')


def validated_stage(plan_path, stage):
    plan = validate_plan(plan_path)
    directory = Path(plan['output']) / stage
    result = campaign.read(directory / 'result.json')
    if (result.get('schema') != SCHEMA + '.stage' or result.get('stage') != stage
            or result['plan'] != campaign.record(plan_path) or result['qualification_eligible'] is not False):
        raise ValueError('executor stage changed its plan')
    claim_path = bound(result['claim'], directory / 'claim.json')
    claim = campaign.read(claim_path)
    if {key: value for key, value in claim.items() if key != 'body_sha256'} != expected_claim(plan_path, stage):
        raise ValueError('executor execution claim changed')
    spec_path = bound(result['specification'], directory / 'worker-spec.json')
    spec = campaign.read(spec_path)
    validate_reconstruction(result['reconstruction'], plan)
    if {key: value for key, value in spec.items() if key != 'body_sha256'} != expected_worker_spec(
            plan_path, stage, result['reconstruction']):
        raise ValueError('executor worker specification changed its exact planned jobs')
    if len(result['results']) != len(spec['jobs']):
        raise ValueError('executor stage has an incomplete result roster')
    for job, row in zip(spec['jobs'], result['results'], strict=True):
        if (row['weight'], row['seed']) != (job['weight'], job['seed']):
            raise ValueError('executor result job identity changed')
        job_directory = Path(job['directory'])
        reference = trainer._seed_reference_path(job_directory,
            trainer.ARCHITECTURES['capacity-12x8'], trainer.ARMS['search-target'], job['seed'])
        bound(row['reference'], reference)
        loaded = trainer._load_seed_receipt_from_reference(job_directory, reference, job['binding'])
        if loaded != row['receipt']:
            raise ValueError('executor check embedded seed receipt changed')
        for key in ('float_checkpoint', 'quantized_runtime'):
            artifact = trainer._output_artifact(job_directory, loaded[key]['path'],
                expected_sha256=loaded[key]['sha256'], label=key)
            bound(row[key], artifact)
            if row[key] != campaign.record(artifact):
                raise ValueError('executor result artifact differs from its receipt-bound bytes')
        source = bound(row['source'], job_directory / f'seed-{job["seed"]}.cpp')
        if source.read_bytes() != adapter._runtime_source(Path(row['quantized_runtime']['path'])):
            raise ValueError('executor check source no longer reproduces its runtime')
    validate_memory(result['memory'])
    evidence = result['process_evidence']
    if stage == 'threads':
        if evidence != []:
            raise ValueError('thread reference cannot claim spawn worker evidence')
    else:
        if len(evidence) != len(spec['jobs']):
            raise ValueError('spawn process evidence has an incomplete seed roster')
        pids = set()
        for job, row, observed in zip(spec['jobs'], result['results'], evidence, strict=True):
            if (observed['weight'], observed['seed']) != (job['weight'], job['seed']) or (
                    observed['reference'] != row['reference']
                    or observed['binding_sha256'] != job['binding']['body_sha256']
                    or observed['native_thread_execution'] != row['receipt']['native_thread_execution']):
                raise ValueError('spawn process evidence changed its seed/reference/native binding')
            values = observed['process']
            if (type(values.get('pid')) is not int or values['pid'] <= 0
                    or values.get('peak_rss_units') != ('bytes' if sys.platform == 'darwin' else 'KiB')
                    or any(not isinstance(values.get(key), (int, float)) or isinstance(values.get(key), bool)
                        or not math.isfinite(values[key]) or values[key] < 0 for key in
                        ('peak_rss', 'minor_page_faults', 'major_page_faults'))):
                raise ValueError('spawn process PID/resource observation is malformed')
            pids.add(values['pid'])
        if not 1 <= len(pids) <= 2:
            raise ValueError('spawn process roster exceeded the two-worker limit')
        if not result['memory']['errors'] and any(str(pid) not in result['memory']['per_pid_peak_rss'] for pid in pids):
            raise ValueError('spawn memory sampler did not observe its executing workers')
    elapsed = result['elapsed_seconds']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('executor elapsed time must be finite and positive')
    return result


def validate_result(plan_path):
    plan = validate_plan(plan_path)
    stages = [validated_stage(plan_path, stage) for stage in STAGES]
    comparison = compare_stage_results(*stages)
    receipt_path = Path(plan['output']) / 'comparison.json'
    body = {'schema': SCHEMA + '.comparison', 'plan': campaign.record(plan_path),
        'stages': [campaign.record(Path(plan['output']) / stage / 'result.json') for stage in STAGES], **comparison}
    campaign.seal(receipt_path, body)
    return body


def run(plan_path):
    plan = validate_plan(plan_path)
    with campaign.lease(Path(plan['root'])):
        for stage in STAGES:
            run_stage(plan_path, stage)
        return validate_result(plan_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    prepare_parser.add_argument('--root', type=Path, required=True)
    prepare_parser.add_argument('--output', type=Path, required=True)
    for name in ('inspect', 'run', 'validate'):
        child = sub.add_parser(name)
        child.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.output)
    elif args.command == 'inspect':
        plan = validate_plan(args.plan)
        result = {'plan': campaign.record(args.plan), 'real_training_started': False,
            'next_command': [sys.executable, str(Path(__file__).resolve()), 'run', '--plan', str(args.plan.resolve())],
            'requires_available_global_heavy_stage_lease': True, 'qualification_eligible': False}
    elif args.command == 'run':
        result = run(args.plan)
    else:
        result = validate_result(args.plan)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
