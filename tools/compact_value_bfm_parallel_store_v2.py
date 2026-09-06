#!/usr/bin/env python3
"""Opt-in bounded, ordered conversion of future exhaustive ranking stores.

No caller of the maintained serial builder is changed. Limits bound raw/shard
bytes, not Python JSON expansion; four and eight workers need separate actual
corpus memory reviews. A group exceeding a configured limit is rejected whole.
Pending-shard limits exclude the growing merged arrays and final group index.
The coordinator retains the serial builder's corpus-sized metadata/seen set,
plus a bounded binary-reader buffer and one whole-group lookahead.
"""
from __future__ import annotations

import argparse
from collections import deque
import concurrent.futures
import contextlib
import fcntl
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import sys

SCHEMA = 'papersoccer.compact-value-bfm-parallel-store.v2'
EXECUTING_SOURCE_BYTES = Path(__file__).read_bytes()
ENV = {name: '1' for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
ENV['PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'] = '1'
_WORKER = None
_LOCK = None


def modules():
    from tools import compact_value_bfm_ranking_store as store
    from tools import compact_value_bfm_training_capacity_v2 as capacity
    expected = Path(__file__).resolve().parent
    for module, filename in ((store, 'compact_value_bfm_ranking_store.py'),
            (store.campaign, 'compact_value_bfm_campaign_v2.py'),
            (store.corpus, 'jacek_replay_corpus.py'), (store.trainer, 'compact_value_bfm_train.py'),
            (capacity, 'compact_value_bfm_training_capacity_v2.py')):
        if Path(module.__file__).resolve() != expected / filename:
            raise ValueError('parallel ranking store requires one exact source namespace')
    package = sys.modules['tools']
    if any(Path(path).resolve() != expected for path in package.__path__):
        raise ValueError('parallel ranking store tools namespace is mixed')
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if name.startswith('tools.') and file and not Path(file).resolve().is_relative_to(expected):
            raise ValueError('parallel ranking store imported a foreign tools provider')
    return store, store.campaign, store.corpus, store.trainer, capacity


def source_closure():
    store, campaign, _corpus, _trainer, _capacity = modules()
    for path, executing in ((Path(__file__), EXECUTING_SOURCE_BYTES),
            (Path(store.__file__), store.EXECUTING_SOURCE_BYTES),
            (Path(campaign.__file__), campaign.EXECUTING_SOURCE_BYTES)):
        if path.read_bytes() != executing:
            raise ValueError('loaded parallel store producer changed on disk')
    return [campaign.record(path) for path in sorted(Path(__file__).resolve().parent.glob('*.py'))]


def settings(workers, max_pending, chunk_bytes, max_group_bytes, max_chunk_output_bytes):
    max_pending = workers if max_pending is None else max_pending
    if type(workers) is not int or not 4 <= workers <= 8:
        raise ValueError('parallel ranking store requires four to eight workers')
    if type(max_pending) is not int or not workers <= max_pending <= 2 * workers:
        raise ValueError('pending chunk bound must be workers through twice workers')
    if any(type(value) is not int or value <= 0 for value in
           (chunk_bytes, max_group_bytes, max_chunk_output_bytes)) or chunk_bytes > max_group_bytes:
        raise ValueError('invalid whole-group chunk byte limits')
    return {'workers': workers, 'max_pending': max_pending, 'chunk_bytes': chunk_bytes,
        'max_group_bytes': max_group_bytes, 'max_chunk_output_bytes': max_chunk_output_bytes,
        'native_threads_per_worker': 1, 'start_method': 'spawn',
        'pending_scratch_bytes_bound': max_pending * (max_group_bytes + max_chunk_output_bytes),
        'parsed_json_memory_bound_claimed': False, 'whole_group_splitting_allowed': False}


def bundle_spec(bundle):
    # Pass maintained bundle fields, not a pickled arbitrary provider object.
    return {'manifest_path': str(Path(bundle.manifest_path).resolve()),
        'manifest': bundle.manifest, 'manifest_payload_hex': bundle.manifest_payload.hex(),
        'body_sha256': bundle.body_sha256}


@contextlib.contextmanager
def lease(root):
    """Resolve the same campaign-wide lease as the maintained stage API."""
    _store, campaign, _corpus, _trainer, _capacity = modules()
    root = Path(root).resolve()
    contract = root / 'campaign.json'
    if contract.exists():
        document = campaign.read(contract)
        if 'heavy_stage_root' in document:
            parent = campaign.verify(document['parent_campaign']).parent
            if Path(document['heavy_stage_root']).resolve() != parent:
                raise ValueError('phase heavy-stage authority changed')
            root = parent
    with (root / '.heavy-stage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock, root


def initialize_worker(claim_record, ticket, lock_identity):
    global _WORKER, _LOCK
    os.environ.update(ENV)
    store, campaign, corpus, trainer, capacity = modules()
    _LOCK = capacity.retain_shared_lock(ticket, lock_identity)
    capacity.parent_death_guard()
    claim = campaign.read(campaign.verify(claim_record))
    if claim['source_closure'] != source_closure():
        raise ValueError('parallel store worker source changed')
    value = claim['bundle']
    bundle = trainer.FrozenBundle(Path(value['manifest_path']),
        bytes.fromhex(value['manifest_payload_hex']), value['manifest'])
    if bundle_spec(bundle) != value:
        raise ValueError('parallel store worker bundle changed')
    _WORKER = claim, bundle


def raw_lines(stream, maximum):
    """Bounded UTF-8 JSONL framing with the serial reader's universal newlines."""
    buffer = b''; ended = False
    while True:
        boundary = min((index for index in (buffer.find(b'\r'), buffer.find(b'\n')) if index >= 0), default=-1)
        if boundary >= 0 and (buffer[boundary:boundary + 1] != b'\r' or boundary + 1 < len(buffer) or ended):
            end = boundary + (2 if buffer[boundary:boundary + 2] == b'\r\n' else 1)
            if end > maximum:
                raise ValueError('whole ranking group exceeds max_group_bytes; operator review and a fresh output with an explicit larger limit required')
            line, buffer = buffer[:boundary] + b'\n', buffer[end:]
            yield line
        elif ended:
            if buffer:
                if len(buffer) > maximum: raise ValueError('whole ranking group exceeds max_group_bytes; operator review required')
                yield buffer
            return
        else:
            if boundary < 0 and len(buffer) > maximum:
                raise ValueError('whole ranking group exceeds max_group_bytes; operator review and a fresh output with an explicit larger limit required')
            block = stream.read(min(65536, maximum + 1))
            buffer += block; ended = not block


def raw_chunks(sources, scratch, limits):
    """At most one lookahead raw line in addition to the bounded pending files."""
    _store, campaign, _corpus, _trainer, _capacity = modules()
    ordinal = 0
    for source in sources:
        opener = gzip.open if source.suffix == '.gz' else open
        with opener(source, 'rb') as stream:
            lines = iter(raw_lines(stream, limits['max_group_bytes']))
            line_number = 0
            pending = None
            while True:
                directory = scratch / f'{ordinal:08d}'
                handle = None; count = 0; total = 0; first_line = line_number + 1
                try:
                    while True:
                        line = pending if pending is not None else next(lines, b'')
                        pending = None
                        if not line: break
                        if len(line) > limits['max_group_bytes']:
                            raise ValueError('whole ranking group exceeds max_group_bytes; operator review and a fresh output with an explicit larger limit required')
                        if count and total + len(line) > limits['chunk_bytes']:
                            pending = line; break
                        if handle is None:
                            directory.mkdir(); handle = (directory / 'raw.jsonl').open('xb')
                        handle.write(line); total += len(line); count += 1; line_number += 1
                        if total >= limits['chunk_bytes']: break
                finally:
                    if handle is not None: handle.close()
                if not count: break
                yield {'ordinal': ordinal, 'source': str(source), 'first_line': first_line,
                    'groups': count, 'raw': campaign.record(directory / 'raw.jsonl'),
                    'directory': str(directory)}
                ordinal += 1


def transcribe(job):
    store, campaign, corpus, trainer, _capacity = modules()
    claim, bundle = _WORKER
    limits = claim['settings']; directory = Path(job['directory'])
    if (directory != Path(claim['output']) / '.scratch' / f"{job['ordinal']:08d}"
            or job['source'] not in [item['path'] for item in claim['sources']]
            or campaign.verify(job['raw']) != directory / 'raw.jsonl'):
        raise ValueError('parallel store shard escaped its frozen claim')
    groups = []; first = None; seen = set(); counts = {'successors': 0, 'active': 0, 'transcripts': 0}
    written = 0
    with trainer.native_thread_execution_scope() as native, contextlib.ExitStack() as stack:
        handles = {name: stack.enter_context((directory / (name + '.part')).open('xb'))
                   for name in ('indices', 'successors', 'transcripts')}
        def write(name, payload):
            nonlocal written
            if written + len(payload) > limits['max_chunk_output_bytes']:
                raise ValueError('whole ranking shard exceeds max_chunk_output_bytes')
            handles[name].write(payload); written += len(payload)
        stream = stack.enter_context(Path(job['raw']['path']).open('rb'))
        for offset, line in enumerate(stream):
            row = corpus.validate_complete_turn_action_group(json.loads(line.decode('utf-8')))
            group = row['group']
            if not group['successors_exhaustive']: raise ValueError('nonexhaustive store group')
            if group['group_id'] in seen: raise ValueError('duplicate/cross-split ranking group')
            seen.add(group['group_id'])
            if row['source_bundle_body_sha256'] != bundle.body_sha256: raise ValueError('ranking bundle changed')
            immutable = {key: row[key] for key in ('feature_schema', 'source_bundle_body_sha256', 'teacher', 'ranking')}
            if first is None:
                first = immutable
                core = {key: row['teacher'][key] for key in ('artifact_sha256', 'payload_sha256', 'feature_schema_sha256')}
                if trainer._validate_teacher_identity(bundle, core) != core: raise ValueError('teacher identity changed')
            elif immutable != first: raise ValueError('mixed ranking identities')
            start = counts['successors']
            for item in group['successors']:
                active = store.np.asarray(item['active'], dtype='<u2'); text = item['transcript'].encode('ascii')
                metadata = store.np.zeros(1, dtype=store.SUCCESSOR_DTYPE)
                metadata['identity'][0] = store.np.void(bytes.fromhex(item['successor_id']))
                metadata['value'][0] = item['teacher_value']; metadata['mover'][0] = item['value_mover']
                metadata['solved'][0] = int(item['proof']['solved'])
                metadata['winner'][0] = -1 if item['proof']['proven_winner'] is None else item['proof']['proven_winner']
                metadata['visits'][0] = item['visits']; metadata['selection_visits'][0] = item['selection_visits']
                metadata['active_begin'][0] = counts['active']; counts['active'] += len(active)
                metadata['active_end'][0] = counts['active']
                metadata['transcript_begin'][0] = counts['transcripts']; counts['transcripts'] += len(text)
                metadata['transcript_end'][0] = counts['transcripts']
                write('indices', active.tobytes()); write('transcripts', text); write('successors', metadata.tobytes())
                counts['successors'] += 1
            entry = {'split': row['split'], 'begin': start, 'end': counts['successors'],
                'group': {key: value for key, value in group.items() if key != 'successors'},
                'source_ordinal': offset, 'source_file': job['source'], 'source_line': job['first_line'] + offset}
            # Include group metadata in the same output budget before retaining it.
            size = len(campaign.raw(entry))
            if written + size > limits['max_chunk_output_bytes']:
                raise ValueError('whole ranking shard exceeds max_chunk_output_bytes')
            written += size; groups.append(entry)
        if first is None or len(groups) != job['groups']: raise ValueError('ranking shard row count changed')
        for handle in handles.values(): handle.flush(); os.fsync(handle.fileno())
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = {'schema': SCHEMA + '.shard', 'job': job, 'identity': first, 'groups': groups, 'counts': counts,
        'arrays': {name: campaign.record(directory / (name + '.part')) for name in handles},
        'pid': os.getpid(), 'native_thread_execution': native,
        'memory': {'peak_rss': usage.ru_maxrss, 'units': 'bytes' if sys.platform == 'darwin' else 'KiB',
                   'minor_page_faults': usage.ru_minflt, 'major_page_faults': usage.ru_majflt}}
    # Constant header/identity overhead is checked too, before publishing the shard.
    payload_bytes = len(campaign.raw({**result,
        'body_sha256': hashlib.sha256(campaign.raw(result)).hexdigest()}))
    array_bytes = sum(item['bytes'] for item in result['arrays'].values())
    if array_bytes + payload_bytes > limits['max_chunk_output_bytes']:
        raise ValueError('whole ranking shard exceeds max_chunk_output_bytes')
    campaign.seal(directory / 'result.json', result)
    return campaign.record(directory / 'result.json')


def ordered_results(pool, jobs, maximum):
    """Never drain a later completion into an unbounded in-memory result list."""
    queue = deque(); jobs = iter(jobs)
    for _ in range(maximum):
        job = next(jobs, None)
        if job is None: break
        queue.append((job, pool.submit(transcribe, job)))
    while queue:
        job, future = queue.popleft()
        yield job, future.result()
        following = next(jobs, None)
        if following is not None: queue.append((following, pool.submit(transcribe, following)))


def merge(job, result_record, handles, state):
    store, campaign, _corpus, _trainer, _capacity = modules()
    result = campaign.read(campaign.verify(result_record))
    directory = Path(job['directory'])
    if (result_record['path'] != str(directory / 'result.json') or result['job'] != job
            or result['schema'] != SCHEMA + '.shard' or job['ordinal'] != state['chunks']):
        raise ValueError('parallel store shard order or identity changed')
    if state['identity'] is None: state['identity'] = result['identity']
    elif state['identity'] != result['identity']: raise ValueError('mixed ranking identities')
    sizes = {'indices': result['counts']['active'] * 2,
             'successors': result['counts']['successors'] * store.SUCCESSOR_DTYPE.itemsize,
             'transcripts': result['counts']['transcripts']}
    paths = {}
    for name, record in result['arrays'].items():
        paths[name] = campaign.verify(record)
        if paths[name] != directory / (name + '.part') or record['bytes'] != sizes[name]:
            raise ValueError('parallel store shard array changed')
    local_end = 0
    for row in result['groups']:
        if (row['begin'] != local_end or row['end'] <= local_end
                or row['end'] > result['counts']['successors']
                or row['source_ordinal'] != len(state['groups']) - state['group_base']
                or row['source_file'] != job['source']
                or row['source_line'] != job['first_line'] + row['source_ordinal']):
            raise ValueError('parallel store shard group coverage changed')
        local_end = row['end']; group_id = row['group']['group_id']
        if group_id in state['seen'] or row['split'] not in ('train', 'validation'):
            raise ValueError('duplicate/cross-split ranking group')
        state['seen'].add(group_id)
        state['groups'].append({**row, 'begin': row['begin'] + state['successors'],
            'end': row['end'] + state['successors'], 'source_ordinal': len(state['groups'])})
    if local_end != result['counts']['successors'] or len(result['groups']) != job['groups']:
        raise ValueError('parallel store shard lost groups')
    for name in ('indices', 'transcripts'):
        with paths[name].open('rb') as source: shutil.copyfileobj(source, handles[name], 1024 * 1024)
    active_end = transcript_end = 0
    with paths['successors'].open('rb') as source:
        while payload := source.read(store.SUCCESSOR_DTYPE.itemsize * 8192):
            metadata = store.np.frombuffer(payload, dtype=store.SUCCESSOR_DTYPE).copy()
            for prefix, previous, total in (('active', active_end, result['counts']['active']),
                    ('transcript', transcript_end, result['counts']['transcripts'])):
                begin, end = metadata[prefix + '_begin'], metadata[prefix + '_end']
                if (int(begin[0]) != previous or store.np.any(begin[1:] != end[:-1])
                        or store.np.any(end <= begin) or int(end[-1]) > total):
                    raise ValueError('parallel store shard offsets changed')
            active_end = int(metadata['active_end'][-1]); transcript_end = int(metadata['transcript_end'][-1])
            for name, offset in (('active_begin', state['active']), ('active_end', state['active']),
                                 ('transcript_begin', state['transcripts']), ('transcript_end', state['transcripts'])):
                if len(metadata) and int(metadata[name].max()) + offset >= 1 << 64:
                    raise ValueError('parallel store offset exceeds uint64')
                metadata[name] += offset
            handles['successors'].write(metadata.tobytes())
    if active_end != result['counts']['active'] or transcript_end != result['counts']['transcripts']:
        raise ValueError('parallel store shard offsets lost data')
    for key in ('successors', 'active', 'transcripts'): state[key] += result['counts'][key]
    state['chunks'] += 1; state['group_base'] = len(state['groups'])
    state['pids'].add(result['pid'])
    previous = state['memory'].get(str(result['pid']))
    if previous is not None and previous['units'] != result['memory']['units']:
        raise ValueError('parallel store worker memory units changed')
    state['memory'][str(result['pid'])] = {'units': result['memory']['units'], **{
        key: max(result['memory'][key], 0 if previous is None else previous[key])
        for key in ('peak_rss', 'minor_page_faults', 'major_page_faults')}}
    state['native'][str(result['pid'])] = result['native_thread_execution']
    state['maximum_raw_chunk_bytes'] = max(state['maximum_raw_chunk_bytes'], job['raw']['bytes'])
    state['maximum_shard_bytes'] = max(state['maximum_shard_bytes'],
        result_record['bytes'] + sum(record['bytes'] for record in result['arrays'].values()))
    shutil.rmtree(directory)


def validate_completed(output, expected):
    store, campaign, _corpus, _trainer, _capacity = modules()
    claim = campaign.read(output / 'parallel-claim.json')
    if {key: value for key, value in claim.items() if key != 'body_sha256'} != expected:
        raise ValueError('parallel ranking-store completed claim changed')
    completion = campaign.read(output / 'parallel-complete.json')
    index = output / 'index.json'; document = campaign.read(index)
    body = {key: value for key, value in document.items() if key not in ('body_sha256', 'parallel_build')}
    if (document.get('parallel_build') != {'claim': campaign.record(output / 'parallel-claim.json'),
            'completion': campaign.record(output / 'parallel-complete.json')}
            or completion['claim'] != campaign.record(output / 'parallel-claim.json')
            or document['sources'] != expected['sources']
            or document['source_bundle_body_sha256'] != expected['bundle']['body_sha256']
            or document['schema'] != store.SCHEMA
            or completion.get('schema') != SCHEMA + '.completion'
            or completion.get('settings') != expected['settings']
            or completion.get('all_submitted_workers_joined') is not True
            or completion.get('sources_reverified') is not True
            or completion.get('store_body_sha256') != hashlib.sha256(campaign.raw(body)).hexdigest()
            or document.get('all_rich_rows_validated') is not True
            or document.get('all_successors_preserved') is not True
            or document.get('protected_tests_opened') is not False):
        raise ValueError('parallel ranking-store completion changed')
    sizes = {'indices': document['active_count'] * 2,
        'successors': document['successor_count'] * store.SUCCESSOR_DTYPE.itemsize,
        'transcripts': document['transcript_bytes']}
    if set(document['arrays']) != set(sizes): raise ValueError('parallel ranking-store array roster changed')
    for name, record in document['arrays'].items():
        path = campaign.verify(record)
        if path != output / (record['sha256'] + f'.{name}.bin') or record['bytes'] != sizes[name]:
            raise ValueError('parallel ranking-store array publication changed')
    if campaign.verify(document['builder']) != output / (campaign.sha(Path(__file__)) + '.builder.py'):
        raise ValueError('parallel ranking-store builder changed')
    if document['arrays'] != completion['arrays']:
        raise ValueError('parallel ranking-store array publication changed')
    return index


def build_store(sources, output, bundle, *, root, workers=4, max_pending=None,
                chunk_bytes=8 * 1024**2, max_group_bytes=64 * 1024**2,
                max_chunk_output_bytes=128 * 1024**2):
    limits = settings(workers, max_pending, chunk_bytes, max_group_bytes, max_chunk_output_bytes)
    root, output = Path(root).resolve(), Path(output).resolve()
    sources = [Path(path).resolve() for path in sources]
    if not output.is_relative_to(root) or output == root or any(path.is_relative_to(output) for path in sources):
        raise ValueError('future ranking-store output must be a separate child of its campaign root')
    os.environ.update(ENV)
    store, campaign, _corpus, _trainer, capacity = modules()
    with lease(root) as (lock, authority):
        expected = {'schema': SCHEMA + '.claim', 'output': str(output), 'root': str(root),
            'heavy_stage_root': str(authority), 'sources': [campaign.record(path) for path in sources],
            'bundle': bundle_spec(bundle), 'settings': limits, 'source_closure': source_closure(),
            'serial_store_schema': store.SCHEMA, 'successor_dtype': repr(store.SUCCESSOR_DTYPE.descr),
            'resume_partial_allowed': False, 'protected_tests_opened': False}
        if output.exists():
            if (output / 'index.json').exists() and (output / 'parallel-complete.json').exists():
                return validate_completed(output, expected)
            raise ValueError('ranking-store output already exists or is partially claimed; use a fresh output')
        output.mkdir(parents=True, exist_ok=False)
        campaign.seal(output / 'parallel-claim.json', expected)
        claim = campaign.record(output / 'parallel-claim.json')
        scratch = output / '.scratch'; scratch.mkdir()
        state = {'identity': None, 'groups': [], 'seen': set(), 'chunks': 0,
                 'successors': 0, 'active': 0, 'transcripts': 0, 'group_base': 0, 'pids': set(),
                 'memory': {}, 'native': {}, 'maximum_raw_chunk_bytes': 0, 'maximum_shard_bytes': 0}
        try:
            with contextlib.ExitStack() as stack:
                handles = {name: stack.enter_context((scratch / (name + '.merged')).open('xb'))
                           for name in ('indices', 'successors', 'transcripts')}
                pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                    mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker,
                    initargs=(claim, capacity.SpawnLockTicket(lock.fileno()), capacity.lock_identity(lock.fileno())))
                try:
                    for job, result in ordered_results(pool, raw_chunks(sources, scratch, limits), limits['max_pending']):
                        merge(job, result, handles, state)
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
                if state['identity'] is None: raise ValueError('empty ranking store')
                for handle in handles.values(): handle.flush(); os.fsync(handle.fileno())
            if expected['source_closure'] != source_closure(): raise ValueError('parallel store producer changed while running')
            for record in expected['sources']: campaign.verify(record)
            arrays = {}
            for name in handles:
                path = scratch / (name + '.merged'); destination = output / (campaign.sha(path) + f'.{name}.bin')
                os.rename(path, destination); arrays[name] = campaign.record(destination)
            builder = output / (campaign.sha(Path(__file__)) + '.builder.py')
            campaign.once(builder, Path(__file__).read_bytes())
            store_body = {'schema': store.SCHEMA, **state['identity'],
                'sources': expected['sources'], 'arrays': arrays, 'groups': state['groups'],
                'successor_count': state['successors'], 'active_count': state['active'],
                'transcript_bytes': state['transcripts'], 'successor_record_bytes': store.SUCCESSOR_DTYPE.itemsize,
                'all_rich_rows_validated': True, 'all_successors_preserved': True,
                'builder': campaign.record(builder), 'protected_tests_opened': False}
            campaign.seal(output / 'parallel-complete.json', {'schema': SCHEMA + '.completion',
                'claim': claim, 'arrays': arrays, 'chunks': state['chunks'], 'worker_pids': sorted(state['pids']),
                'store_body_sha256': hashlib.sha256(campaign.raw(store_body)).hexdigest(),
                'all_submitted_workers_joined': True, 'sources_reverified': True, 'settings': limits,
                'worker_memory_observations': state['memory'], 'memory_observations_simultaneous': False,
                'worker_native_thread_execution': state['native'],
                'maximum_raw_chunk_bytes': state['maximum_raw_chunk_bytes'],
                'maximum_shard_bytes': state['maximum_shard_bytes'], 'capacity_approval_claimed': False})
            completion = campaign.record(output / 'parallel-complete.json')
            campaign.seal(output / 'index.json', {**store_body,
                'parallel_build': {'claim': claim, 'completion': completion}})
            return output / 'index.json'
        except BaseException as error:
            campaign.seal(output / 'parallel-failure.json', {'schema': SCHEMA + '.failure', 'claim': claim,
                'type': type(error).__name__, 'message': str(error), 'resume_partial_allowed': False})
            raise
        finally:
            shutil.rmtree(scratch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'output', 'bundle-manifest'): parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--source', action='append', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=range(4, 9), default=4)
    for name, default in (('chunk-bytes', 8 * 1024**2), ('max-group-bytes', 64 * 1024**2),
                          ('max-chunk-output-bytes', 128 * 1024**2)):
        parser.add_argument('--' + name, type=int, default=default)
    args = parser.parse_args(); os.environ.update(ENV)
    _store, _campaign, _corpus, trainer, _capacity = modules()
    # Bundle parsing is metadata-only; actual source conversion owns the lease.
    bundle = trainer.FrozenBundle.load(args.bundle_manifest)
    print(build_store(args.source, args.output, bundle, root=args.root, workers=args.workers,
        chunk_bytes=args.chunk_bytes, max_group_bytes=args.max_group_bytes,
        max_chunk_output_bytes=args.max_chunk_output_bytes))


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
