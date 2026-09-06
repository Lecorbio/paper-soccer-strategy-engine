#!/usr/bin/env python3
"""Fresh real-smoke equivalence for a private float-ranking decision cache.

A stdlib-only supervisor runs FA and the explicit candidate in separate fresh
namespaces. Both execute nine refined-adaptive seeds through the maintained
spawn runner. Only smoke reconstruction, the diagnostic initializer, and a
read-only call observer adapt that runner. No phase, optimizer recipe, numerical
receipt field, qualification, or production activation is changed by this tool.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib
import importlib.machinery
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import threading
from types import ModuleType, SimpleNamespace

ID = 'compact-value-bfm-trained-v2'
SCHEMA = ID + '.validation-cache-check.v2'
REFERENCE_COMMIT = 'fa012e7783ae374b64f18d884d5e563794fbdf9c'
PROFILE = 'refined-adaptive-scales-v1'
WEIGHTS = (0., .1, .25)
SEEDS = (20260907, 20260908, 20260909)
ENGINES = ('reference', 'candidate')
ENV = {name: '1' for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                             'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
ENV.update({'PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
POLICY = {'qat_profile': PROFILE, 'weights': list(WEIGHTS), 'seeds': list(SEEDS),
    'cohort_order': list(ENGINES), 'maximum_seed_processes': 4, 'numerical_threads_per_seed': 1,
    'native_games': 64, 'fresh_seeds_per_engine': 9, 'historical_seed_results_reused': False,
    'global_heavy_stage_lease_required': True, 'concurrent_pipeline_allowed': False,
    'incomplete_claim_retry_allowed': False, 'binding_differences_allowed': [],
    'native_provenance_differences_allowed': [], 'require_cached_mapped_path': True,
    'require_float_ranking_forward_reduction_per_seed': True,
    'observer': 'same-read-only-profile-hook-outside-numerical-receipts-v1',
    'wall_time_speedup_claim_allowed': False, 'qualification_eligible': False,
    'production_activation_allowed': False, 'protected_or_live_work_allowed': False}
_ENGINE = None
_WORKER_LOCK = None


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def record(path):
    path = Path(path).resolve()
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def verify(binding, expected=None):
    path = Path(binding['path']).absolute()
    if path.resolve() != path or expected is not None and path != Path(expected).resolve() or record(path) != binding:
        raise ValueError('cache equivalence artifact/source path or hash changed')
    return path


def read(path):
    value = json.loads(Path(path).read_bytes()); body = dict(value); digest = body.pop('body_sha256')
    if hashlib.sha256(raw(body)).hexdigest() != digest:
        raise ValueError('cache equivalence receipt hash changed')
    return value


def seal(path, body):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = raw({**body, 'body_sha256': hashlib.sha256(raw(body)).hexdigest()})
    if path.exists():
        if path.read_bytes() != payload: raise ValueError('immutable cache equivalence artifact differs')
        return record(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    with temporary.open('xb') as file:
        file.write(payload); file.flush(); os.fsync(file.fileno())
    try:
        try: os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload: raise ValueError('cache equivalence publication raced')
    finally:
        temporary.unlink()
    return record(path)


def output_path(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    parent = root / 'diagnostics/validation-cache-equivalence'
    if output == parent or not output.is_relative_to(parent):
        raise ValueError('cache equivalence output must be an explicit fresh diagnostic child')
    return output


def snapshot(root, path, name):
    root = Path(root).resolve(); path = Path(path).resolve(); document = read(path)
    commit = document.get('commit', '')
    if (len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit)
            or path != root / 'source-snapshots' / commit / 'snapshot.json'
            or document.get('schema') != ID + '.source-snapshot.v2'
            or document.get('repository') != str(path.parent / 'repository')
            or (name == 'reference') != (commit == REFERENCE_COMMIT)):
        raise ValueError('reference must be FA and candidate a separate explicit immutable snapshot')
    verify(document['archive'], path.parent / 'source.tar')
    return {'commit': commit, 'repository': document['repository'], 'snapshot': record(path), 'archive': document['archive']}


def bootstrap(repository):
    """No numerical provider is loaded in the supervising interpreter."""
    global _ENGINE
    repository = Path(repository).resolve(); expected = repository / 'tools'
    if _ENGINE is not None:
        if _ENGINE.repository != repository: raise ValueError('cannot mix numerical engine namespaces')
        return _ENGINE
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if name == 'tools' and any(Path(path).resolve() != expected for path in module.__path__):
            raise ValueError('cache equivalence engine requires a fresh interpreter')
        if file and Path(file).resolve() != Path(__file__).resolve() and (
                name.startswith(('tools.', 'teacher_training_')) or Path(file).stem.startswith(('compact_value_bfm_', 'jacek_replay_'))):
            if not Path(file).resolve().is_relative_to(repository):
                raise ValueError('foreign numerical provider in cache equivalence interpreter')
    sys.path[:] = [path for path in sys.path if not (Path(path or os.getcwd()) / 'tools').is_dir()
                  or Path(path or os.getcwd()).resolve() == repository]
    sys.path.insert(0, str(repository)); os.environ.update(ENV)
    if 'tools' not in sys.modules:
        package = ModuleType('tools'); package.__package__ = 'tools'; package.__path__ = [str(expected)]
        package.__spec__ = importlib.machinery.ModuleSpec('tools', loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__; sys.modules['tools'] = package
    names = {'campaign': 'compact_value_bfm_campaign_v2', 'check': 'compact_value_bfm_seed_process_check_v2',
        'process': 'compact_value_bfm_seed_process_v2', 'trainer': 'compact_value_bfm_train',
        'store': 'compact_value_bfm_ranking_store', 'capacity': 'compact_value_bfm_training_capacity_v2',
        'resources': 'compact_value_bfm_training_resources_v2'}
    values = {key: importlib.import_module('tools.' + name) for key, name in names.items()}
    for key, module in values.items():
        if Path(module.__file__).resolve() != expected / (names[key] + '.py'):
            raise ValueError('cache equivalence loaded another engine provider')
    _ENGINE = SimpleNamespace(repository=repository, **values)
    return _ENGINE


def refined_binding(base, profile_contract):
    """Only the named QAT settings differ from the historical standard recipe."""
    body = copy.deepcopy(base); body.pop('body_sha256')
    if body['settings']['qat_profile'] != 'standard-v1' or profile_contract.get('qat_profile') != PROFILE:
        raise ValueError('real-smoke binding/profile is not the approved refined recipe')
    body['settings']['qat_profile'] = PROFILE
    body['settings']['qat_profile_contract'] = profile_contract
    return {**body, 'body_sha256': hashlib.sha256(raw(body)).hexdigest()}


def metadata(root, engine_descriptor):
    engine = bootstrap(engine_descriptor['repository'])
    value = engine.check.smoke_inputs_metadata(Path(root))
    profile = engine.trainer.qat_profile_contract(PROFILE)
    sources = engine.check.sources()
    with tarfile.open(verify(engine_descriptor['archive']), 'r:') as archive:
        for binding in sources:
            path = verify(binding); relative = path.relative_to(engine.repository)
            member = archive.getmember(relative.as_posix())
            if not member.isfile() or archive.extractfile(member).read() != path.read_bytes():
                raise ValueError('numerical source differs from its committed snapshot archive')
    return {'schema': SCHEMA + '.engine-metadata', 'engine': engine_descriptor,
        'sources': sources, 'smoke': value, 'profile_contract': profile,
        'refined_base_binding': refined_binding(value['expected_base_binding'], profile),
        'runtime': {'python': str(Path(sys.executable).resolve()), 'version': sys.version,
                    'numpy': engine.trainer.np.__version__},
        'has_cache_factory': callable(getattr(engine.trainer, '_new_float_ranking_decision_cache', None)),
        'metadata_reconstructed_core_inputs': False, 'metadata_trained_seeds': False}


def _metadata_process(root, engine, output):
    request = output.parent / (output.stem + '-request.json')
    seal(request, {'root': str(Path(root).resolve()), 'engine': engine, 'output': str(output), 'harness': record(__file__)})
    command = [sys.executable, str(Path(__file__).resolve()), '_metadata', '--request', str(request)]
    result = subprocess.run(command, capture_output=True, timeout=600, env={**os.environ, **ENV})
    if result.returncode:
        raise ValueError('metadata-only engine verification failed: ' + result.stderr.decode(errors='replace')[-2000:])
    return record(output)


def prepare(root, candidate_snapshot, output, *, after_driver_launch, resource_authorization=None):
    root, output = Path(root).resolve(), output_path(root, output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('cache equivalence requires a fresh output; existing claims/receipts stay unchanged')
    driver_path = root / 'drivers/attempt-003-data-through-workspace/launch.json'
    driver = record(after_driver_launch); verify(driver, driver_path)
    authorization = record(resource_authorization or root / 'training-resources/more-cores-authorization.json')
    engines = {'reference': snapshot(root, root / 'source-snapshots' / REFERENCE_COMMIT / 'snapshot.json', 'reference'),
               'candidate': snapshot(root, candidate_snapshot, 'candidate')}
    metadata_records = {name: _metadata_process(root, descriptor, output / 'metadata' / (name + '.json'))
                        for name, descriptor in engines.items()}
    documents = [read(verify(metadata_records[name])) for name in ENGINES]
    left, right = documents
    if (left['smoke'] != right['smoke'] or left['profile_contract'] != right['profile_contract']
            or left['refined_base_binding'] != right['refined_base_binding'] or left['runtime'] != right['runtime']
            or left['has_cache_factory'] is not False or right['has_cache_factory'] is not True):
        raise ValueError('engine metadata differs beyond the explicit cache implementation')
    plan = {'schema': SCHEMA + '.plan', 'root': str(root), 'output': str(output), 'harness': record(__file__),
        'engines': engines, 'metadata': metadata_records, 'expected_binding': left['refined_base_binding'],
        'policy': POLICY, 'runtime': left['runtime'], 'after_driver_launch': driver,
        'training_resource_authorization': authorization, 'heavy_comparison_started': False}
    binding = seal(output / 'plan.json', plan)
    validate_plan(verify(binding))
    return binding


def validate_plan(path):
    path = Path(path).resolve(); plan = read(path); root = Path(plan['root'])
    if (plan.get('schema') != SCHEMA + '.plan' or plan.get('policy') != POLICY
            or plan.get('harness') != record(__file__) or plan.get('heavy_comparison_started') is not False
            or output_path(root, plan['output']) / 'plan.json' != path):
        raise ValueError('cache comparison source, scope or plan path changed')
    if set(plan['engines']) != set(ENGINES) or set(plan['metadata']) != set(ENGINES):
        raise ValueError('cache comparison engine roster changed')
    documents = []
    for name in ENGINES:
        descriptor = plan['engines'][name]
        if snapshot(root, verify(descriptor['snapshot']), name) != descriptor:
            raise ValueError('cache comparison immutable engine snapshot changed')
        document = read(verify(plan['metadata'][name], path.parent / 'metadata' / (name + '.json')))
        if (document['engine'] != descriptor or document['runtime'] != plan['runtime']
                or document['refined_base_binding'] != plan['expected_binding']
                or document['metadata_reconstructed_core_inputs'] is not False or document['metadata_trained_seeds'] is not False):
            raise ValueError('cache comparison engine/input metadata changed')
        for source in document['sources']:
            verify(source)
        if refined_binding(document['smoke']['expected_base_binding'], document['profile_contract']) != plan['expected_binding']:
            raise ValueError('cache comparison changed substantive training bindings')
        documents.append(document)
    if documents[0]['smoke'] != documents[1]['smoke'] or documents[0]['has_cache_factory'] is not False or documents[1]['has_cache_factory'] is not True:
        raise ValueError('cache comparison does not bind one corpus and an actual cache candidate')
    verify(plan['after_driver_launch'], root / 'drivers/attempt-003-data-through-workspace/launch.json')
    verify(plan['training_resource_authorization'], root / 'training-resources/more-cores-authorization.json')
    return plan


def engine_for_plan(plan_path, name):
    plan = validate_plan(plan_path)
    if name not in ENGINES: raise ValueError('unknown cache comparison engine')
    engine = bootstrap(plan['engines'][name]['repository'])
    expected = read(verify(plan['metadata'][name]))
    if metadata(plan['root'], plan['engines'][name]) != {key: value for key, value in expected.items() if key != 'body_sha256'}:
        raise ValueError('engine no longer reproduces its separately frozen source/input metadata')
    engine.resources.validate_authorization(plan['training_resource_authorization'], Path(plan['root']))
    return engine, plan


def mapped_inputs(engine, inputs):
    result = {}
    for split in ('train', 'validation'):
        groups = getattr(inputs.successor_rankings, split)
        if not isinstance(groups, tuple) or not groups:
            raise ValueError('equivalence requires the actual immutable mapped smoke group tuples')
        for group in groups:
            if type(group.successors) is not engine.store.MappedSuccessors or type(group.successors.store) is not engine.store.RankingStore:
                raise ValueError('equivalence cannot substitute eager/mutable successor containers')
            for name in ('metadata', 'indices', 'transcripts'):
                array = getattr(group.successors.store, name)
                if not isinstance(array, engine.trainer.np.memmap) or array.mode != 'r' or array.flags.writeable:
                    raise ValueError('equivalence mapped successor backing must remain read-only')
        result[split] = {'container': 'tuple', 'groups': len(groups), 'successors': sum(len(group.successors) for group in groups),
                         'mapped_successors': True, 'all_backings_readonly_memmap': True}
    return result


def reconstruct(engine, plan, name):
    smoke = read(verify(plan['metadata'][name]))['smoke']
    bundle, inputs, identity = engine.check.reconstruct_smoke({'smoke': smoke})
    actual = engine.trainer.training_binding(bundle, inputs, engine.trainer.ARCHITECTURES['capacity-12x8'],
        engine.trainer.ARMS['search-target'], SEEDS[0], None, WEIGHTS[0],
        verify(smoke['initial_checkpoint']), PROFILE)
    if actual != plan['expected_binding']:
        raise ValueError('engine recomputation changed the complete refined training binding')
    return bundle, inputs, identity, mapped_inputs(engine, inputs)


def jobs(plan, name, engine):
    return [{'weight': weight, 'seed': seed,
             'directory': str(Path(plan['output']) / name / 'training' / f'lambda-{weight:.2f}'),
             'binding': engine.process.roster_binding(plan['expected_binding'], seed, weight)}
            for weight in WEIGHTS for seed in SEEDS]


def worker_spec(plan_path, name, identity, mapped, engine):
    plan = read(plan_path)
    return {'schema': SCHEMA + '.worker', 'plan': record(plan_path), 'engine_name': name,
        'executor': {'mode': 'spawn-v2', 'maximum_workers': 4}, 'ranking_weights': list(WEIGHTS),
        'seeds': list(SEEDS), 'qat_profile': PROFILE, 'jobs': jobs(plan, name, engine),
        'reconstruction': identity, 'mapped_inputs': mapped, 'policy': POLICY}


def initialize_worker(spec_path, lock_ticket, expected_lock):
    global _WORKER_LOCK
    spec = read(spec_path); plan_path = verify(spec['plan'])
    engine, plan = engine_for_plan(plan_path, spec['engine_name'])
    _WORKER_LOCK = engine.capacity.retain_shared_lock(lock_ticket, expected_lock)
    engine.capacity.parent_death_guard()
    with engine.trainer.native_thread_execution_scope():
        bundle, inputs, identity, mapped = reconstruct(engine, plan, spec['engine_name'])
    if spec != {**worker_spec(plan_path, spec['engine_name'], identity, mapped, engine), 'body_sha256': spec['body_sha256']}:
        raise ValueError('cache comparison worker specification changed')
    verify(record(spec_path), Path(plan['output']) / spec['engine_name'] / 'worker-spec.json')
    initial = read(verify(plan['metadata'][spec['engine_name']]))['smoke']['initial_checkpoint']
    engine.process._WORKER = (spec, bundle, inputs, verify(initial))


def observed_run(job):
    """Observe calls in the maintained runner's one-seed thread; never wrap math."""
    engine = _ENGINE
    if engine is None or job not in engine.process._WORKER[0]['jobs']:
        raise ValueError('observed seed is outside its source-bound roster')
    trainer = engine.trainer
    factory = getattr(trainer, '_new_float_ranking_decision_cache', None)
    stats = {'ranking_metrics_calls': 0, 'float_ranking_forward_calls': 0, 'quantized_ranking_forward_calls': 0,
        'other_float_forward_calls': 0, 'other_quantized_forward_calls': 0,
        'cache_factory_calls': 0, 'immutable_mapped_factory_returns': 0}
    def observe(frame, event, value):
        if event == 'call' and frame.f_code is trainer.successor_ranking_metrics.__code__:
            stats['ranking_metrics_calls'] += 1
        if event == 'call' and frame.f_code is trainer.forward.__code__:
            quantized = frame.f_locals.get('quantized') is not None
            ranking = frame.f_back is not None and frame.f_back.f_code is trainer.successor_ranking_metrics.__code__
            key = ('quantized_' if quantized else 'float_') + 'ranking_forward_calls' if ranking else (
                'other_quantized_forward_calls' if quantized else 'other_float_forward_calls')
            stats[key] += 1
        if factory is not None and event == 'return' and frame.f_code is factory.__code__:
            stats['cache_factory_calls'] += 1
            if value is None or value.immutable_mapped_groups is not True:
                raise ValueError('real-seed cache fell back from the immutable mapped path')
            stats['immutable_mapped_factory_returns'] += 1
    previous = threading.getprofile()
    if previous is not None:
        raise ValueError('a preexisting observer would confound cache execution evidence')
    threading.setprofile(observe)
    try:
        result = engine.process._run(job)
    finally:
        threading.setprofile(previous)
    return {**result, 'cache_observation': {'observer': record(__file__), 'policy': POLICY['observer'],
        'profile_hook_restored': threading.getprofile() is previous, 'counts': stats}}


class ObservedPool:
    """Reuse the maintained flattened roster/collection/shutdown unchanged."""
    def __init__(self, pool, original): self.pool, self.original = pool, original
    def map(self, function, iterable):
        if function is not self.original: raise ValueError('unexpected maintained seed dispatch function')
        return self.pool.map(observed_run, iterable)
    def shutdown(self, **kwargs): return self.pool.shutdown(**kwargs)


def run_cohort(plan_path, name, lock_fd):
    engine, plan = engine_for_plan(plan_path, name); directory = Path(plan['output']) / name
    require_driver_finished(plan)
    if any((directory / value).exists() for value in ('worker-spec.json', 'training', 'result.json')):
        raise ValueError('cohort requires fresh outputs; completed or interrupted seeds cannot substitute')
    with engine.trainer.native_thread_execution_scope():
        bundle, inputs, identity, mapped = reconstruct(engine, plan, name)
    spec_path = directory / 'worker-spec.json'; seal(spec_path, worker_spec(plan_path, name, identity, mapped, engine))
    capacity = engine.capacity
    class Executor(engine.process.SpawnSeedExecutor):
        def __enter__(self):
            self.pool = ObservedPool(__import__('concurrent.futures', fromlist=['ProcessPoolExecutor']).ProcessPoolExecutor(
                max_workers=4, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker,
                initargs=(str(self.path), capacity.SpawnLockTicket(lock_fd), capacity.lock_identity(lock_fd))), engine.process._run)
            return self
    with Executor(spec_path) as executor:
        receipts = executor.run_roster()
        rows = [engine.check.export_result(job, receipt)
                for job, receipt in zip(executor.spec['jobs'], receipts, strict=True)]
        evidence = executor.evidence
    binding = seal(directory / 'result.json', {'schema': SCHEMA + '.cohort', 'plan': record(plan_path), 'engine_name': name,
        'metadata': plan['metadata'][name], 'worker_spec': record(spec_path), 'reconstruction': identity,
        'mapped_inputs': mapped, 'results': rows, 'process_evidence': evidence, 'policy': POLICY,
        'qualification_eligible': False})
    return binding


def validate_observation(value):
    if (value.get('observer') != record(__file__) or value.get('policy') != POLICY['observer']
            or value.get('profile_hook_restored') is not True):
        raise ValueError('cache call observer provenance or restoration changed')
    expected = {'ranking_metrics_calls', 'float_ranking_forward_calls', 'quantized_ranking_forward_calls',
                'other_float_forward_calls', 'other_quantized_forward_calls', 'cache_factory_calls', 'immutable_mapped_factory_returns'}
    counts = value['counts']
    if set(counts) != expected or any(type(number) is not int or number < 0 for number in counts.values()):
        raise ValueError('cache call counts are not complete nonnegative integer evidence')
    return counts



def validate_mapped_summary(summary, binding):
    if set(summary) != {'train', 'validation'}:
        raise ValueError('cohort mapped-input summary omitted a split')
    for split, value in summary.items():
        if (set(value) != {'container', 'groups', 'successors', 'mapped_successors', 'all_backings_readonly_memmap'}
                or value['container'] != 'tuple' or value['mapped_successors'] is not True
                or value['all_backings_readonly_memmap'] is not True or type(value['groups']) is not int
                or value['groups'] != binding['successor_ranking'][split + '_groups']
                or type(value['successors']) is not int or value['successors'] < value['groups'] or value['groups'] <= 0):
            raise ValueError('cohort mapped-input evidence differs from its whole training binding')

def validate_cohort(plan_path, name):
    engine, plan = engine_for_plan(plan_path, name); directory = Path(plan['output']) / name
    result = read(directory / 'result.json'); spec = read(verify(result['worker_spec'], directory / 'worker-spec.json'))
    if (result.get('schema') != SCHEMA + '.cohort' or result['plan'] != record(plan_path)
            or result['engine_name'] != name or result['metadata'] != plan['metadata'][name]
            or result['policy'] != POLICY or result['qualification_eligible'] is not False
            or spec != {**worker_spec(plan_path, name, result['reconstruction'], result['mapped_inputs'], engine), 'body_sha256': spec['body_sha256']}):
        raise ValueError('cohort source/input/roster binding changed')
    validate_mapped_summary(result['mapped_inputs'], plan['expected_binding'])
    engine.check.validate_reconstruction(result['reconstruction'], {'smoke': read(verify(plan['metadata'][name]))['smoke']})
    if len(result['results']) != 9 or len(result['process_evidence']) != 9:
        raise ValueError('cohort requires all nine fresh completed real seeds')
    pids = set()
    for job, row, observed in zip(spec['jobs'], result['results'], result['process_evidence'], strict=True):
        if (row['weight'], row['seed']) != (job['weight'], job['seed']): raise ValueError('cohort seed order changed')
        reference = engine.trainer._seed_reference_path(Path(job['directory']), engine.trainer.ARCHITECTURES['capacity-12x8'],
                                                       engine.trainer.ARMS['search-target'], job['seed'])
        verify(row['reference'], reference)
        loaded = engine.trainer._load_seed_receipt_from_reference(Path(job['directory']), reference, job['binding'])
        if loaded != row['receipt'] or loaded['binding'] != job['binding']:
            raise ValueError('cohort seed changed its whole substantive binding or receipt')
        for key in ('float_checkpoint', 'quantized_runtime'):
            expected = engine.trainer._output_artifact(Path(job['directory']), loaded[key]['path'],
                expected_sha256=loaded[key]['sha256'], label=key)
            verify(row[key], expected)
        source = verify(row['source'], Path(job['directory']) / f'seed-{job["seed"]}.cpp')
        if source.read_bytes() != engine.check.adapter._runtime_source(verify(row['quantized_runtime'])):
            raise ValueError('cohort source bytes do not reproduce the exact runtime')
        if ((observed['weight'], observed['seed']) != (job['weight'], job['seed'])
                or observed['reference'] != row['reference'] or observed['binding_sha256'] != job['binding']['body_sha256']
                or observed['native_thread_execution'] != loaded['native_thread_execution']):
            raise ValueError('cohort process evidence lost seed/reference/native bindings')
        engine.trainer.validate_native_thread_execution(observed['native_thread_execution'])
        validate_observation(observed['cache_observation'])
        pid = observed['process']['pid']
        if type(pid) is not int or pid <= 0: raise ValueError('cohort child PID is malformed')
        pids.add(pid)
    if not 1 <= len(pids) <= 4: raise ValueError('cohort exceeded its four-process seed-worker bound')
    return result


def compare_results(plan, reference, candidate):
    if raw(reference['reconstruction']) != raw(candidate['reconstruction']) or raw(reference['mapped_inputs']) != raw(candidate['mapped_inputs']):
        raise ValueError('engines reconstructed different complete inputs or mapped group containers')
    expected = [(weight, seed) for weight in WEIGHTS for seed in SEEDS]
    if any([(row['weight'], row['seed']) for row in result['results']] != expected
           or len(result['process_evidence']) != len(expected) for result in (reference, candidate)):
        raise ValueError('engine comparison requires the exact nine-seed roster')
    checks = []
    for left, right, left_process, right_process in zip(reference['results'], candidate['results'],
            reference['process_evidence'], candidate['process_evidence'], strict=True):
        if (left['weight'], left['seed']) != (right['weight'], right['seed']):
            raise ValueError('engine comparison seed identities differ')
        a, b = left['receipt'], right['receipt']
        if raw(a['binding']) != raw(b['binding']):
            raise ValueError('whole substantive training bindings differ; none may be stripped')
        if raw(a['native_thread_execution']) != raw(b['native_thread_execution']):
            raise ValueError('native execution provenance differs; no unexplained difference is authorized')
        if raw({key: value for key, value in a.items() if key not in ('body_sha256', 'native_thread_execution')}) != raw({
                key: value for key, value in b.items() if key not in ('body_sha256', 'native_thread_execution')}):
            raise ValueError('deterministic seed receipt/trial/metric fields differ')
        for key in ('float_checkpoint', 'quantized_runtime', 'source'):
            if (left[key]['sha256'], left[key]['bytes']) != (right[key]['sha256'], right[key]['bytes']):
                raise ValueError('float checkpoint/runtime/source bytes differ')
        x, y = validate_observation(left_process['cache_observation']), validate_observation(right_process['cache_observation'])
        if (x['cache_factory_calls'] != 0 or x['immutable_mapped_factory_returns'] != 0
                or y['cache_factory_calls'] <= 0 or y['immutable_mapped_factory_returns'] != y['cache_factory_calls']
                or not 0 < y['float_ranking_forward_calls'] < x['float_ranking_forward_calls']
                or any(x[key] != y[key] for key in ('ranking_metrics_calls', 'quantized_ranking_forward_calls',
                                                  'other_float_forward_calls', 'other_quantized_forward_calls'))):
            raise ValueError('candidate failed to demonstrate exact mapped cache use and reduced float ranking work')
        checks.append({'weight': left['weight'], 'seed': left['seed'], 'entire_binding_equal': True,
            'all_deterministic_receipt_fields_equal': True, 'exact_artifact_bytes_equal': True,
            'native_execution_provenance_equal_and_validated': True,
            'reference_float_ranking_forwards': x['float_ranking_forward_calls'],
            'candidate_float_ranking_forwards': y['float_ranking_forward_calls'],
            'immutable_mapped_cache_factory_calls': y['cache_factory_calls']})
    if len(checks) != 9: raise ValueError('equivalence requires all nine matched seeds')
    return {'schema': SCHEMA + '.comparison', 'policy': POLICY, 'engines': plan['engines'],
        'checks': checks, 'exact_equivalence_passed': True, 'mapped_cache_exercised': True,
        'production_activation_allowed': False, 'qualification_eligible': False, 'campaign_success': False}


def require_driver_finished(plan):
    launch = read(verify(plan['after_driver_launch'])); root = Path(plan['root'])
    expected = launch['process']
    current = subprocess.run(['ps', '-p', str(expected['pid']), '-o', 'args='], capture_output=True, text=True)
    if current.returncode == 0 and current.stdout.strip() == expected['command']:
        raise ValueError('active data/capacity/workspace driver must exit before heavy equivalence')
    path = root / 'drivers/attempt-003-data-through-workspace/result.json'
    result = read(path)
    if (result.get('completed') is not True or result.get('failure') is not None
            or result.get('plan') != launch['plan'] or result.get('producer') != launch['driver']
            or result.get('production_training_started') is not False or result.get('optimizer_steps') != 0):
        raise ValueError('data/capacity/workspace driver lacks a completed source-bound stop-before-training result')
    verify(launch['plan']); verify(launch['driver'])
    return record(path)


def parent_watch(fd):
    try:
        while os.read(fd, 1): pass
    finally:
        if os.getpgrp() == os.getpid(): os.killpg(os.getpid(), signal.SIGKILL)


def _owned_command(plan_path, name, mode, output, lock_fd=None, watch_fd=None):
    command = [sys.executable, str(Path(__file__).resolve()), mode, '--plan', str(Path(plan_path).resolve()),
               '--engine', name, '--output', str(output)]
    if lock_fd is not None: command += ['--lock-fd', str(lock_fd), '--parent-watch-fd', str(watch_fd)]
    return command


def subprocess_cohort(plan_path, name, lock_fd):
    plan = read(plan_path); directory = Path(plan['output']) / name
    if directory.exists(): raise ValueError('cohort was already claimed; incomplete work is never retried')
    read_fd, write_fd = os.pipe(); process = None; error = None
    output = directory / 'result.json'; command = _owned_command(plan_path, name, '_cohort', output, lock_fd, read_fd)
    seal(directory / 'claim.json', {'schema': SCHEMA + '.cohort-claim', 'plan': record(plan_path),
        'engine_name': name, 'command': command, 'policy': POLICY, 'retry_allowed': False})
    try:
        with (directory / 'stdout.log').open('xb') as out, (directory / 'stderr.log').open('xb') as err:
            process = subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True,
                pass_fds=(read_fd, lock_fd), env={**os.environ, **ENV})
            os.close(read_fd); read_fd = None
            seal(directory / 'process.json', {'schema': SCHEMA + '.owned-cohort-process',
                'claim': record(directory / 'claim.json'), 'pid': process.pid, 'process_group': process.pid})
            code = process.wait(timeout=86400)
            if code != 0: raise ValueError('owned equivalence cohort failed; preserve its spent claim')
    except BaseException as failure:
        error = {'type': type(failure).__name__, 'message': str(failure)}
        raise
    finally:
        if process is not None:
            if process.poll() is None or process.returncode != 0:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            process.wait()
        if read_fd is not None: os.close(read_fd)
        os.close(write_fd)
        seal(directory / 'execution.json', {'schema': SCHEMA + '.cohort-execution', 'claim': record(directory / 'claim.json'),
            'returncode': None if process is None else process.returncode, 'failure': error,
            'cohort_result': record(output) if output.exists() else None,
            'stdout': record(directory / 'stdout.log') if (directory / 'stdout.log').exists() else None,
            'stderr': record(directory / 'stderr.log') if (directory / 'stderr.log').exists() else None})


def validate_result(plan_path):
    plan = validate_plan(plan_path); outputs = []
    for name in ENGINES:
        directory = Path(plan['output']) / name
        execution = read(directory / 'execution.json')
        if execution['returncode'] != 0 or execution['failure'] is not None:
            raise ValueError('both engines need successful fresh cohort executions')
        claim = read(verify(execution['claim'], directory / 'claim.json'))
        command = claim.get('command', [])
        try:
            lock_fd = int(command[command.index('--lock-fd') + 1])
            watch_fd = int(command[command.index('--parent-watch-fd') + 1])
        except (ValueError, IndexError, TypeError) as error:
            raise ValueError('cohort execution lost its owned lease/watch descriptors') from error
        if (claim.get('schema') != SCHEMA + '.cohort-claim' or claim.get('plan') != record(plan_path)
                or claim.get('engine_name') != name or claim.get('policy') != POLICY
                or claim.get('retry_allowed') is not False or min(lock_fd, watch_fd) < 0
                or command != _owned_command(plan_path, name, '_cohort', directory / 'result.json', lock_fd, watch_fd)):
            raise ValueError('cohort execution command/source claim changed')
        owner = read(directory / 'process.json')
        if (owner.get('schema') != SCHEMA + '.owned-cohort-process'
                or owner.get('claim') != record(directory / 'claim.json')
                or type(owner.get('pid')) is not int or owner['pid'] <= 0 or owner.get('process_group') != owner['pid']):
            raise ValueError('cohort process-group provenance changed')
        verify(execution['cohort_result'], directory / 'result.json')
        for key in ('stdout', 'stderr'): verify(execution[key], directory / (key + '.log'))
        validated = directory / 'validated.json'
        command = _owned_command(plan_path, name, '_validate-cohort', validated)
        result = subprocess.run(command, capture_output=True, timeout=600, env={**os.environ, **ENV})
        if result.returncode: raise ValueError('independent engine-specific cohort validation failed: ' + result.stderr.decode(errors='replace')[-2000:])
        proof = read(validated)
        if (proof.get('schema') != SCHEMA + '.independent-validation'
                or proof.get('cohort') != record(directory / 'result.json') or proof.get('plan') != record(plan_path)
                or proof.get('engine') != name or proof.get('engine_metadata') != plan['metadata'][name]
                or proof.get('all_nine_seed_bindings_and_artifacts_validated') is not True):
            raise ValueError('independent cohort verification belongs to another source/plan')
        outputs.append(read(directory / 'result.json'))
    body = compare_results(plan, *outputs)
    body.update({'plan': record(plan_path), 'cohorts': [record(Path(plan['output']) / name / 'result.json') for name in ENGINES]})
    return seal(Path(plan['output']) / 'comparison.json', body)


def run(plan_path):
    plan = validate_plan(plan_path); root = Path(plan['root']); directory = Path(plan['output'])
    require_driver_finished(plan)
    with (root / '.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require_driver_finished(plan)
        if any((directory / name).exists() for name in ('claim.json', 'result.json', 'reference', 'candidate')):
            raise ValueError('equivalence run is already claimed; validate completed results, never retry incomplete work')
        seal(directory / 'claim.json', {'schema': SCHEMA + '.run-claim', 'plan': record(plan_path), 'policy': POLICY,
            'driver_completion': require_driver_finished(plan), 'retry_allowed': False})
        failure = None; result = None
        try:
            for name in ENGINES: subprocess_cohort(plan_path, name, lock.fileno())
            result = validate_result(plan_path)
            return result
        except BaseException as error:
            failure = {'type': type(error).__name__, 'message': str(error)}; raise
        finally:
            seal(directory / 'result.json', {'schema': SCHEMA + '.run-result', 'plan': record(plan_path),
                'claim': record(directory / 'claim.json'), 'comparison': result, 'failure': failure,
                'completed': result is not None and failure is None, 'policy': POLICY,
                'production_activation_allowed': False, 'campaign_success': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'validate', '_metadata', '_cohort', '_validate-cohort'))
    for name in ('root', 'candidate-snapshot', 'output', 'after-driver-launch', 'resource-authorization', 'plan', 'request'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--engine', choices=ENGINES); parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--parent-watch-fd', type=int)
    args = parser.parse_args(); os.environ.update(ENV)
    if args.command == 'prepare':
        if not all((args.root, args.candidate_snapshot, args.output, args.after_driver_launch)):
            parser.error('prepare requires root/candidate-snapshot/output/after-driver-launch')
        result = prepare(args.root, args.candidate_snapshot, args.output, after_driver_launch=args.after_driver_launch,
                         resource_authorization=args.resource_authorization)
    elif args.command == '_metadata':
        request = read(args.request); verify(request['harness'], Path(__file__).resolve())
        result = seal(request['output'], metadata(request['root'], request['engine']))
    elif args.command in ('_cohort', '_validate-cohort'):
        if args.plan is None or args.engine is None or args.output is None:
            parser.error('engine command requires plan/engine/output')
        if args.command == '_cohort':
            if args.lock_fd is None or args.parent_watch_fd is None or os.getpgrp() != os.getpid():
                parser.error('cohort requires its owned process group, lease and parent watch')
            threading.Thread(target=parent_watch, args=(args.parent_watch_fd,), daemon=True).start()
            plan = validate_plan(args.plan)
            expected_lock = (Path(plan['root']) / '.heavy-stage.lock').stat(); inherited = os.fstat(args.lock_fd)
            if (inherited.st_dev, inherited.st_ino) != (expected_lock.st_dev, expected_lock.st_ino):
                raise ValueError('cohort inherited another heavy-stage lease')
            fcntl.flock(args.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require_driver_finished(plan)
            claim = read(Path(plan['output']) / args.engine / 'claim.json')
            if (claim.get('schema') != SCHEMA + '.cohort-claim' or claim.get('plan') != record(args.plan)
                    or claim.get('engine_name') != args.engine or claim.get('policy') != POLICY
                    or claim.get('retry_allowed') is not False
                    or claim.get('command') != _owned_command(args.plan, args.engine, '_cohort',
                        Path(plan['output']) / args.engine / 'result.json', args.lock_fd, args.parent_watch_fd)):
                raise ValueError('owned cohort lost its source-bound launch claim')
            result = run_cohort(args.plan, args.engine, args.lock_fd)
        else:
            value = validate_cohort(args.plan, args.engine)
            result = seal(args.output, {'schema': SCHEMA + '.independent-validation', 'plan': record(args.plan),
                'cohort': record(Path(read(args.plan)['output']) / args.engine / 'result.json'), 'engine': args.engine,
                'engine_metadata': read(args.plan)['metadata'][args.engine], 'all_nine_seed_bindings_and_artifacts_validated': True})
    else:
        if args.plan is None: parser.error('run/validate requires a plan')
        result = run(args.plan) if args.command == 'run' else validate_result(args.plan)
    print(raw(result).decode(), end='', flush=True)


if __name__ == '__main__': main()
