#!/usr/bin/env python3
"""Allocation-only workspace measurement using an explicitly frozen fa012e7 engine.

This file is a standalone bootstrap, deliberately separate from the frozen tools
package. No optimizer, candidate training, quality test, or automatic activation
is performed. Real forward/backward arrays have extended lifetimes for a
conservative measurement; their expression temporaries remain transient.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gc
import hashlib
import heapq
import importlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import resource
import sys
from types import ModuleType, SimpleNamespace

ENGINE_COMMIT = 'fa012e7783ae374b64f18d884d5e563794fbdf9c'
SCHEMA = 'compact-value-bfm-trained-v2.training-workspace.v2'
ENVIRONMENT = {name: '1' for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
MARKER = 'PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'
POLICY = {'workers': 4, 'start_method': 'spawn', 'numerical_threads_per_worker': 1,
    'training_groups': 'largest-K-all-standard-train-groups;K=ceil(groups/ceil(new_rows/64));group-id-ties',
    'validation_fixture': 'largest-real-validation-group-repeated;two-view-tuples-three-forward-caches',
    'output_gradients': 'allocation-only-zero-filled-float32;at-most-nine-nonzero-rows-per-group',
    'backward_retention': 'successful-return-profile-hook-on-exact-frozen-code;shallow-local-copy',
    'training_validation_scalar_fixtures_held_together': True,
    'original_float_and_deployed_scales_only': True, 'optimizer_steps': 0,
    'quality_test': False, 'training_artifacts_written': False,
    'automatic_production_activation': False, 'production_peak_headroom_proven': False}


def raw(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def record(path):
    path = Path(path).resolve()
    with path.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def verified(binding):
    if record(binding['path']) != binding:
        raise ValueError('workspace source/input artifact changed')
    return Path(binding['path'])


def sealed(path):
    value = json.loads(Path(path).read_bytes()); body = dict(value)
    claimed = body.pop('body_sha256')
    if hashlib.sha256(raw(body)).hexdigest() != claimed:
        raise ValueError('workspace bootstrap receipt hash changed')
    return value


def isolate_engine_namespace(engine_root):
    """Never repair an already mixed interpreter by replacing loaded modules."""
    engine_root = Path(engine_root).resolve(); expected = engine_root / 'tools'
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if name == 'tools':
            paths = list(getattr(module, '__path__', []))
            if not paths or any(Path(path).resolve() != expected for path in paths):
                raise ValueError('workspace requires a fresh interpreter; tools package is not the frozen engine')
        elif name.startswith(('tools.', 'teacher_training_')) or (file and Path(file).stem.startswith(('compact_value_bfm_', 'jacek_replay_'))):
            if file and Path(file).resolve() == Path(__file__).resolve():
                if name.startswith('tools.'):
                    raise ValueError('run the new workspace probe standalone, outside the frozen tools namespace')
                continue
            if file and not Path(file).resolve().is_relative_to(engine_root):
                raise ValueError('workspace found a repository module from another engine')
    # Keep only the engine's tools namespace discoverable in spawned interpreters.
    sys.path[:] = [value for value in sys.path if not (Path(value or os.getcwd()) / 'tools').is_dir()
                   or Path(value or os.getcwd()).resolve() == engine_root]
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    probe_directory = str(Path(__file__).resolve().parent)
    if probe_directory not in sys.path:
        sys.path.append(probe_directory)
    if 'tools' not in sys.modules:
        package = ModuleType('tools'); package.__package__ = 'tools'; package.__path__ = [str(expected)]
        package.__spec__ = importlib.machinery.ModuleSpec('tools', loader=None, is_package=True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules['tools'] = package


def bootstrap(capacity_plan):
    capacity_path = verified(capacity_plan) if isinstance(capacity_plan, dict) else Path(capacity_plan).resolve()
    plan = sealed(capacity_path)
    root = Path(plan['root']).resolve()
    engine_root = root / 'source-snapshots' / ENGINE_COMMIT / 'repository'
    sources = {Path(item['path']).name: item for item in plan['sources']}
    required = ('compact_value_bfm_training_capacity_v2.py', 'compact_value_bfm_seed_process_v2.py',
                'compact_value_bfm_train.py', 'compact_value_bfm_campaign_v2.py', 'compact_value_bfm_ranking_store.py',
                'compact_value_bfm_teacher_training.py')
    for name in required:
        if name not in sources or verified(sources[name]).resolve() != engine_root / 'tools' / name:
            raise ValueError('capacity plan does not bind the required fa012e7 numerical engine')
    isolate_engine_namespace(engine_root)
    os.environ.update(ENVIRONMENT); os.environ[MARKER] = '1'
    capacity = importlib.import_module('tools.compact_value_bfm_training_capacity_v2')
    campaign, trainer, resources, sampling = capacity.modules()
    process = importlib.import_module('tools.compact_value_bfm_seed_process_v2')
    store = importlib.import_module('tools.compact_value_bfm_ranking_store')
    adapter = importlib.import_module('tools.compact_value_bfm_teacher_training')
    for module in (capacity, campaign, trainer, process, store, adapter):
        if record(module.__file__) != sources[Path(module.__file__).name]:
            raise ValueError('loaded numerical provider differs from the frozen source record')
    provider_sources = {}
    for name, filename in (('model_exporter', 'export_model.py'), ('source_exporter', 'export_submission.py'), ('gate_support', 'rank4_gate_support.py')):
        module = getattr(adapter, name)
        expected = engine_root / 'submissions/codingame/bots/compact_value_bfm' / filename
        if Path(module.__file__).resolve() != expected:
            raise ValueError('loaded adapter dependency is outside the frozen engine')
        provider_sources[name] = record(expected)
    isolate_engine_namespace(engine_root)
    capacity.validate_plan(capacity_path)
    return SimpleNamespace(root=root, capacity=capacity, campaign=campaign, trainer=trainer,
                           process=process, store=store, sampling=sampling, sources=plan['sources'], provider_sources=provider_sources)


def validation_envelope(engine, capacity_plan):
    """Only the bound JSON group index is read; no CSR or mapped feature arrays."""
    audit = engine.campaign.read(verified(capacity_plan['input_audit']))
    index = engine.campaign.read(verified(audit['ranking_store']))
    groups = [row for row in index['groups'] if row['split'] == 'validation']
    if not groups: raise ValueError('workspace index has no real validation group')
    chosen = min(groups, key=lambda row: (-(row['end'] - row['begin']), row['group']['group_id']))
    return {'ranking_store': audit['ranking_store'], 'validation_group_id': chosen['group']['group_id'],
            'V': chosen['end'] - chosen['begin'], 'validation_groups': len(groups)}


def prepare(capacity_plan, output, *, hold_seconds=10, reconstruction_timeout_seconds=3600):
    engine = bootstrap(capacity_plan); c = engine.campaign
    capacity_path = verified(capacity_plan) if isinstance(capacity_plan, dict) else Path(capacity_plan).resolve()
    base = engine.capacity.validate_plan(capacity_path)
    completed = engine.capacity.validate_result(capacity_path)
    if completed['memory_measurements_complete'] is not True:
        raise ValueError('workspace requires completed source-bound load-only memory evidence')
    allowed = engine.root / 'diagnostics/training-workspace'; output = Path(output).resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('workspace output must be a fresh child of diagnostics/training-workspace')
    if type(hold_seconds) is not int or not 2 <= hold_seconds <= 60 or type(reconstruction_timeout_seconds) is not int or not 1 <= reconstruction_timeout_seconds <= 7200:
        raise ValueError('workspace hold/timeout must remain bounded')
    contract = c.read(verified(base['phase_contract']))
    inputs = {key: contract['inputs'][key] for key in ('attempt_one_initial_checkpoint', 'attempt_zero_runtime')}
    for item in inputs.values(): verified(item)
    body = {'schema': SCHEMA + '.plan', 'capacity_plan': record(capacity_path),
        'capacity_result': record(Path(base['output']) / 'result.json'), 'output': str(output),
        'root': str(engine.root), 'context': base['context'], 'phase': base['phase'], 'inputs': inputs,
        'engine_commit': ENGINE_COMMIT, 'engine_sources': engine.sources, 'adapter_dependency_sources': engine.provider_sources,
        'probe_source': record(__file__),
        'runtime': engine.capacity.runtime(), 'numpy_version': engine.trainer.np.__version__, 'policy': POLICY,
        'expected_input_identity': completed['coordinator']['identity'],
        'training_envelope': completed['workspace_dimensions_metadata_only'],
        'validation_envelope': validation_envelope(engine, base),
        'hold_seconds': hold_seconds, 'reconstruction_timeout_seconds': reconstruction_timeout_seconds,
        'metadata_only_preparation': True}
    c.seal(output / 'plan.json', body)
    return record(output / 'plan.json')


def validate_plan(path):
    path = Path(path).resolve(); plan = sealed(path)
    if plan.get('probe_source') != record(__file__) or plan.get('policy') != POLICY or plan.get('engine_commit') != ENGINE_COMMIT:
        raise ValueError('workspace probe source/policy/engine changed')
    engine = bootstrap(plan['capacity_plan']); c = engine.campaign
    base = engine.capacity.validate_plan(verified(plan['capacity_plan']))
    completed = engine.capacity.validate_result(verified(plan['capacity_plan']))
    allowed = engine.root / 'diagnostics/training-workspace'
    if (Path(plan['output']).resolve() / 'plan.json' != path or Path(plan['output']) == allowed
            or not Path(plan['output']).is_relative_to(allowed)
            or plan['capacity_result'] != record(Path(base['output']) / 'result.json')
            or plan['engine_sources'] != engine.sources or plan['adapter_dependency_sources'] != engine.provider_sources
            or plan['runtime'] != engine.capacity.runtime()
            or plan['numpy_version'] != engine.trainer.np.__version__
            or plan['expected_input_identity'] != completed['coordinator']['identity']
            or plan['training_envelope'] != completed['workspace_dimensions_metadata_only']
            or plan['validation_envelope'] != validation_envelope(engine, base)
            or completed['memory_measurements_complete'] is not True
            or any(plan[key] != base[key] for key in ('root', 'context', 'phase'))
            or plan.get('metadata_only_preparation') is not True
            or type(plan['hold_seconds']) is not int or not 2 <= plan['hold_seconds'] <= 60
            or type(plan['reconstruction_timeout_seconds']) is not int or not 1 <= plan['reconstruction_timeout_seconds'] <= 7200):
        raise ValueError('workspace plan lost its completed actual-corpus capacity binding')
    contract = c.read(verified(base['phase_contract']))
    if plan['inputs'] != {key: contract['inputs'][key] for key in plan['inputs']} or set(plan['inputs']) != {'attempt_one_initial_checkpoint', 'attempt_zero_runtime'}:
        raise ValueError('workspace changed original checkpoint or frozen deployed scales')
    for item in plan['inputs'].values(): verified(item)
    return engine, plan


def select_groups(trainer, inputs):
    groups = inputs.successor_rankings.train
    if not groups or any(trainer._ranking_group_profile(group) != 'standard-v1' for group in groups):
        raise ValueError('workspace top-K envelope requires standard unweighted training groups')
    batches = math.ceil(len(inputs.new) / trainer.NEW_ROWS_PER_BATCH)
    if not batches or len(inputs.new) < 64 or len(inputs.anchor) < 192:
        raise ValueError('workspace corpus cannot supply the real64/192 scalar fixture')
    count = math.ceil(len(groups) / batches)
    selected = heapq.nsmallest(count, groups, key=lambda group: (-len(group.successors), group.group_id))
    validation = inputs.successor_rankings.validation
    if not validation:
        raise ValueError('workspace requires a real held-out validation group')
    largest = min(validation, key=lambda group: (-len(group.successors), group.group_id))
    return selected, largest


def selection_summary(selected, validation):
    return {'training_groups': [{'group_id': group.group_id, 'successors': len(group.successors)} for group in selected],
            'N': sum(len(group.successors) for group in selected), 'K': len(selected),
            'validation_group_id': validation.group_id, 'V': len(validation.successors)}


def validate_selection(summary, plan):
    envelope = plan['training_envelope']; rows = summary['training_groups']
    if (envelope['unweighted_all_groups_upper_bound_used'] is not True
            or type(summary['K']) is not int or summary['K'] != envelope['ranking_groups_per_batch_upper_bound']
            or type(summary['N']) is not int or summary['N'] != envelope['successors_per_batch_conservative_upper_bound']
            or len(rows) != summary['K'] or len({row['group_id'] for row in rows}) != len(rows)
            or any(type(row['successors']) is not int or row['successors'] <= 0 for row in rows)
            or sum(row['successors'] for row in rows) != summary['N']
            or rows != sorted(rows, key=lambda row: (-row['successors'], row['group_id']))
            or type(summary['V']) is not int or summary['V'] <= 0
            or not isinstance(summary['validation_group_id'], str)
            or summary['V'] != plan['validation_envelope']['V']
            or summary['validation_group_id'] != plan['validation_envelope']['validation_group_id']
            or summary['validation_group_id'] in {row['group_id'] for row in rows}):
        raise ValueError('workspace selection differs from the frozen all-group allocation envelope')


def retain_backward(function, *args):
    """Capture actual successful-return locals without replacing numerical code."""
    previous = sys.getprofile(); captured = []
    def profile(frame, event, result):
        if previous is not None: previous(frame, event, result)
        if frame.f_code is function.__code__ and event == 'return' and isinstance(result, dict):
            captured.append(dict(frame.f_locals))
    try:
        sys.setprofile(profile)
        gradients = function(*args)
    finally:
        sys.setprofile(previous)
    if len(captured) != 1 or set(gradients) != {'w1', 'w2', 'w3'}:
        raise ValueError('workspace did not capture exactly one successful frozen backward return')
    return gradients, captured[0]


def allocation_fixture(trainer, inputs, parameters, quantized, selected, validation):
    np = trainer.np
    before = {key: trainer._array_identity(value) for key, value in parameters.items()}
    scalar_active = (*inputs.new.active_rows(range(64)), *inputs.anchor.active_rows(range(192)))
    scalar_output, scalar_cache = trainer.forward(parameters, trainer.ARCHITECTURES['capacity-12x8'], scalar_active, quantized=quantized)
    active = tuple(successor.active for group in selected for successor in group.successors)
    output, cache = trainer.forward(parameters, trainer.ARCHITECTURES['capacity-12x8'], active, quantized=quantized)
    derivative = np.empty(len(active), dtype=np.float32); derivative.fill(0)
    offset = 0; nonzero = 0
    for group in selected:
        count = min(9, len(group.successors)); derivative[offset:offset + count] = np.float32(.01)
        nonzero += count; offset += len(group.successors)
    effective = quantized.effective()
    gradients, backward = retain_backward(trainer._network_gradients, parameters,
        trainer.ARCHITECTURES['capacity-12x8'], active, cache, derivative, effective)
    previous_active = tuple(successor.active for successor in validation.successors)
    previous_float = trainer.forward(parameters, trainer.ARCHITECTURES['capacity-12x8'], previous_active)
    previous_quantized = trainer.forward(parameters, trainer.ARCHITECTURES['capacity-12x8'], previous_active, quantized=quantized)
    next_active = tuple(successor.active for successor in validation.successors)
    next_float = trainer.forward(parameters, trainer.ARCHITECTURES['capacity-12x8'], next_active)
    if before != {key: trainer._array_identity(value) for key, value in parameters.items()}:
        raise ValueError('allocation-only workspace changed original model parameters')
    retained = {'scalar_active': scalar_active, 'scalar_output': scalar_output, 'scalar_cache': scalar_cache,
        'training_active': active, 'training_output': output, 'training_cache': cache,
        'artificial_derivative': derivative, 'effective': effective, 'gradients': gradients,
        'backward_locals': backward, 'previous_validation_active': previous_active,
        'previous_float': previous_float, 'previous_quantized': previous_quantized,
        'next_validation_active': next_active, 'next_float': next_float}
    arrays = {key: {'shape': list(value.shape), 'dtype': str(value.dtype), 'bytes': value.nbytes}
              for key, value in backward.items() if isinstance(value, np.ndarray)}
    return retained, {'artificial_nonzero_output_rows': nonzero, 'zero_gradient_array_fully_written': True,
        'parameters_unchanged': True, 'backward_named_arrays': arrays,
        'all_training_successors_forwarded': len(active), 'two_validation_view_tuples': True,
        'validation_forward_cache_sets': 3, 'profile_hook_restored': True}


def mapped_ranges(selected, validation):
    groups = [*selected, validation]; store = groups[0].successors.store
    ranges = {key: [] for key in ('indices', 'transcripts')}
    for group in groups:
        successors = group.successors
        if successors.store is not store:
            raise ValueError('workspace groups do not share the bound ranking store')
        first, last = store.metadata[successors.begin], store.metadata[successors.end - 1]
        ranges['indices'].append((int(first['active_begin']) * 2, int(last['active_end']) * 2))
        ranges['transcripts'].append((int(first['transcript_begin']), int(last['transcript_end'])))
    coverage = {}
    for name, spans in ranges.items():
        merged = []
        for begin, end in sorted(spans):
            if merged and begin <= merged[-1][1]: merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
            else: merged.append((begin, end))
        coverage[name] = sum(end - begin for begin, end in merged)
    arrays = store.document['arrays']
    return {'mapped_artifacts': arrays, 'total_extent_bytes': sum(item['bytes'] for item in arrays.values()),
        'selected_index_transcript_range_bytes': coverage,
        'outside_selected_index_transcript_ranges_bytes': sum(arrays[key]['bytes'] - coverage[key] for key in coverage),
        'metadata_scanned_during_reconstruction': True, 'range_coverage_is_not_pinned_residency': True,
        'same_selected_ranges_in_all_four_workers': True}


def usage():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return {'ru_maxrss': value.ru_maxrss, 'ru_maxrss_units': 'bytes' if sys.platform == 'darwin' else 'KiB',
            'minor_page_faults': value.ru_minflt, 'major_page_faults': value.ru_majflt,
            'ru_maxrss_is_process_lifetime_peak': True}


def validate_ranges(ranges, plan):
    arrays = plan['training_envelope']['mapped_array_artifacts']
    coverage = ranges['selected_index_transcript_range_bytes']
    if (ranges['mapped_artifacts'] != arrays or ranges['total_extent_bytes'] != sum(row['bytes'] for row in arrays.values())
            or set(coverage) != {'indices', 'transcripts'}
            or any(type(value) is not int or not 0 <= value <= arrays[key]['bytes'] for key, value in coverage.items())
            or ranges['outside_selected_index_transcript_ranges_bytes'] != sum(arrays[key]['bytes'] - coverage[key] for key in coverage)
            or ranges['metadata_scanned_during_reconstruction'] is not True
            or ranges['range_coverage_is_not_pinned_residency'] is not True
            or ranges['same_selected_ranges_in_all_four_workers'] is not True):
        raise ValueError('workspace mapped-extent accounting changed its source or residency limitations')


def worker(plan_path, ordinal, connection, lock_ticket, expected_lock):
    engine = bootstrap(sealed(plan_path)['capacity_plan']); cap = engine.capacity
    owned_lock = cap.retain_shared_lock(lock_ticket, expected_lock); cap.parent_death_guard()
    bundle = inputs = retained = parameters = quantized = None
    try:
        _engine, plan = validate_plan(plan_path)
        with engine.trainer.native_thread_execution_scope() as native:
            bundle, inputs, identity = engine.process.reconstruct_inputs(Path(plan['context']), plan['phase'])
            if identity != plan['expected_input_identity']: raise ValueError('workspace reconstruction differs from completed capacity')
            selected, validation = select_groups(engine.trainer, inputs)
            summary = selection_summary(selected, validation)
            validate_selection(summary, plan)
            parameters = engine.trainer.load_float_checkpoint(verified(plan['inputs']['attempt_one_initial_checkpoint']), engine.trainer.ARCHITECTURES['capacity-12x8'])
            arch, deployed, _selection, _doc = engine.trainer.load_runtime(verified(plan['inputs']['attempt_zero_runtime']))
            quantized = engine.trainer.quantize_fixed(parameters, arch, deployed.scales)
            before = usage()
            retained, allocations = allocation_fixture(engine.trainer, inputs, parameters, quantized, selected, validation)
            after = usage()
            connection.send({'kind': 'ready', 'ordinal': ordinal, 'pid': os.getpid(), 'identity': identity,
                'native_thread_execution': native, 'runtime': cap.runtime(), 'lock': cap.lock_identity(owned_lock),
                'workspace': {'selection': summary, 'allocations': allocations, 'mapped_ranges': mapped_ranges(selected, validation),
                              'usage_before_workspace': before, 'usage_after_workspace': after}})
            cap.retain_until_release(connection, plan['reconstruction_timeout_seconds'] + plan['hold_seconds'])
    except BaseException as error:
        cap.clear_failed_frames(error)
        try: connection.send({'kind': 'error', 'ordinal': ordinal, 'pid': os.getpid(), 'error': type(error).__name__ + ': ' + str(error)})
        except (OSError, EOFError): pass
        raise
    finally:
        bundle = inputs = retained = parameters = quantized = None
        if 'selected' in locals(): selected = validation = None
        gc.collect(); connection.close(); os.close(owned_lock)


def run(path):
    engine, plan = validate_plan(path); c, cap = engine.campaign, engine.capacity
    output = Path(plan['output']); bundle = inputs = None
    with (engine.root / '.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output / name).exists() for name in ('claim.json', 'result.json', 'failure.json')):
            raise ValueError('workspace execution is spent; preserve its result or failure')
        c.seal(output / 'claim.json', {'schema': SCHEMA + '.claim', 'plan': record(path), 'policy': POLICY,
            'pid': os.getpid(), 'global_lock': cap.lock_identity(lock.fileno()), 'started_at_utc': cap.utc_now(), 'retry_allowed': False})
        try:
            with engine.sampling.MemorySampler() as sampler:
                with engine.trainer.native_thread_execution_scope() as native:
                    bundle, inputs, identity = engine.process.reconstruct_inputs(Path(plan['context']), plan['phase'])
                    if identity != plan['expected_input_identity']: raise ValueError('coordinator inputs changed from completed capacity')
                    selected, validation = select_groups(engine.trainer, inputs)
                    expected = selection_summary(selected, validation)
                    validate_selection(expected, plan)
                    ranges = mapped_ranges(selected, validation)
                    validate_ranges(ranges, plan)
                    cohort = cap.hold_four(path, plan, identity, sampler, lock.fileno(), worker_target=worker)
                    if any(row['workspace']['selection'] != expected or row['workspace']['mapped_ranges'] != ranges for row in cohort['children']):
                        raise ValueError('four workspace workers selected different actual-corpus envelopes')
                    for row in cohort['children']: validate_allocations(row['workspace']['allocations'], expected)
            memory = sampler.report()
            if (memory['errors'] or memory['samples'] < 2 or any(str(pid) not in memory['per_pid_peak_rss']
                    for pid in [os.getpid(), *(row['pid'] for row in cohort['children'])])):
                raise ValueError('workspace memory sampler did not observe all five retained owners')
            return c.seal(output / 'result.json', {'schema': SCHEMA + '.result', 'plan': record(path),
                'claim': record(output / 'claim.json'), 'completed_at_utc': cap.utc_now(), 'policy': POLICY,
                'coordinator': {'pid': os.getpid(), 'identity': identity, 'native_thread_execution': native},
                'cohort': cohort, 'memory': memory, 'selection': expected, 'mapped_ranges': ranges,
                'limits': ['Preparation and epoch/report allocation peaks are not measured.',
                          'Backward named arrays are retained beyond normal return, but expression temporaries and BLAS internals are not retained.',
                          'ru_maxrss includes earlier reconstruction peaks and cannot be reset to a workspace-only peak.',
                          'Ranking candidate/pair-selection allocations and Adam moments/update temporaries are not executed.',
                          'All four workers use the same largest ranges; actual distinct batches can touch more shared file pages.'],
                'production_peak_headroom_proven': False, 'automatic_production_activation': False,
                'optimizer_steps': 0, 'training_artifacts_written': False, 'quality_test': False})
        except BaseException as error:
            cap.clear_failed_frames(error)
            c.seal(output / 'failure.json', {'schema': SCHEMA + '.failure', 'plan': record(path),
                'claim': record(output / 'claim.json'), 'failed_at_utc': cap.utc_now(), 'retry_allowed': False,
                'error': type(error).__name__ + ': ' + str(error)})
            raise
        finally:
            bundle = inputs = None
            if 'selected' in locals(): selected = validation = None
            gc.collect()


def validate(path):
    engine, plan = validate_plan(path); c, cap = engine.campaign, engine.capacity
    output = Path(plan['output']); result = c.read(output / 'result.json'); claim = c.read(output / 'claim.json')
    if (result['schema'] != SCHEMA + '.result' or result['plan'] != record(path) or result['claim'] != record(output / 'claim.json')
            or result['policy'] != POLICY or claim['policy'] != POLICY or claim['plan'] != result['plan']
            or claim['retry_allowed'] is not False or result['optimizer_steps'] != 0
            or any(result[key] is not False for key in ('production_peak_headroom_proven', 'automatic_production_activation', 'training_artifacts_written', 'quality_test'))):
        raise ValueError('workspace result changed its allocation-only scope')
    cohort = result['cohort']; children = cohort['children']; identity = result['coordinator']['identity']
    if (len(children) != 4 or [row['ordinal'] for row in children] != list(range(4))
            or len({row['pid'] for row in children}) != 4
            or any(type(row['pid']) is not int or row['pid'] <= 0 or row['pid'] == claim['pid'] for row in children)
            or cohort['all_four_ready_and_held'] is not True or cohort['child_exitcodes'] != [0]*4
            or cohort['minimum_hold_seconds'] != plan['hold_seconds'] or identity != plan['expected_input_identity']
            or result['coordinator']['pid'] != claim['pid']):
        raise ValueError('workspace result lost its four-worker retained-input ownership')
    validate_selection(result['selection'], plan)
    validate_ranges(result['mapped_ranges'], plan)
    for row in [result['coordinator'], *children]: engine.trainer.validate_native_thread_execution(row['native_thread_execution'])
    for row in children:
        if (row['identity'] != identity or row['lock'] != claim['global_lock'] or row['runtime'] != cap.runtime()
                or row['workspace']['selection'] != result['selection'] or row['workspace']['mapped_ranges'] != result['mapped_ranges']
                or row['workspace']['allocations']['parameters_unchanged'] is not True
                or row['workspace']['allocations']['profile_hook_restored'] is not True):
            raise ValueError('workspace child evidence changed source/model/retention binding')
        validate_allocations(row['workspace']['allocations'], result['selection'])
        before, after = row['workspace']['usage_before_workspace'], row['workspace']['usage_after_workspace']
        for reading in (before, after):
            if (reading['ru_maxrss_units'] != ('bytes' if sys.platform == 'darwin' else 'KiB')
                    or reading['ru_maxrss_is_process_lifetime_peak'] is not True
                    or any(type(reading[key]) not in (int, float) or not math.isfinite(reading[key]) or reading[key] < 0
                           for key in ('ru_maxrss', 'minor_page_faults', 'major_page_faults'))):
                raise ValueError('workspace process resource observation is malformed')
        if any(after[key] < before[key] for key in ('ru_maxrss', 'minor_page_faults', 'major_page_faults')):
            raise ValueError('workspace lifetime resource counters moved backwards')
    engine.sampling.validate_memory(result['memory'])
    memory = result['memory']
    if (memory['errors'] or memory['samples'] < 2
            or any(str(pid) not in memory['per_pid_peak_rss'] for pid in [claim['pid'], *(row['pid'] for row in children)])):
        raise ValueError('workspace memory sampler did not observe all five retained owners')
    return result


def validate_allocations(allocations, summary):
    n = summary['N']
    required = {'output_gradient': [n], 'output_pre_gradient': [n], 'second_gradient': [n, 8],
        'second_pre_gradient': [n, 8], 'first_gradient': [n, 12], 'first_pre_gradient': [n, 12],
        'first_pre': [n, 12], 'first': [n, 12], 'second_pre': [n, 8], 'second': [n, 8], 'output_pre': [n]}
    if (allocations['all_training_successors_forwarded'] != n
            or allocations['artificial_nonzero_output_rows'] != sum(min(9, row['successors']) for row in summary['training_groups'])
            or allocations['zero_gradient_array_fully_written'] is not True
            or allocations['two_validation_view_tuples'] is not True
            or allocations['validation_forward_cache_sets'] != 3):
        raise ValueError('workspace fixture changed its real row counts or conservative retention')
    for name, shape in required.items():
        if allocations['backward_named_arrays'].get(name) != {'shape': shape, 'dtype': 'float32', 'bytes': 4 * math.prod(shape)}:
            raise ValueError('workspace did not retain the required dense frozen backward arrays')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--capacity-plan', type=Path, required=True); prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--hold-seconds', type=int, default=10); prep.add_argument('--reconstruction-timeout-seconds', type=int, default=3600)
    for name in ('run', 'validate'): sub.add_parser(name).add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare': result = prepare(args.capacity_plan, args.output, hold_seconds=args.hold_seconds, reconstruction_timeout_seconds=args.reconstruction_timeout_seconds)
    else: result = run(args.plan) if args.command == 'run' else validate(args.plan)
    print(json.dumps(result if args.command == 'prepare' else {'selection': result['selection'], 'optimizer_steps': 0,
                                                            'production_peak_headroom_proven': False}), flush=True)


if __name__ == '__main__': main()
