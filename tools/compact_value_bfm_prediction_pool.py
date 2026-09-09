"""Opt-in scalar-validation predictors. No optimizer, ranking or label work.

The parent owns metrics and blocks until every helper finishes each generation.
All model packets are immutable bytes before they enter asynchronous queues.
"""
from __future__ import annotations

import hashlib
import importlib
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import resource
import sys
import threading
import time

ENV = {name: '1' for name in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
MARKER = 'PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY'
SCHEMA = 'papersoccer.compact-value-bfm-validation-prediction.v1'
MAX_FEATURE_BYTES = 256 * 1024 * 1024
MAX_MODEL_BYTES = 1024 * 1024
MAX_CALLS = 4096
BATCH_ROWS = 4096
TIMEOUT_SECONDS = 60
_SOURCE_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_BUDGET_LOCK = threading.Lock()
_RESERVED = 0


class PredictionError(ValueError):
    pass


def _trainer():
    return importlib.import_module('tools.compact_value_bfm_train')


def policy(workers=1, seed_concurrency=1):
    if type(workers) is not int or workers not in (1, 2, 4):
        raise PredictionError('validation workers must be exactly1,2or4')
    if type(seed_concurrency) is not int or seed_concurrency not in (1, 2, 4):
        raise PredictionError('seed concurrency must be a supported actual roster size')
    if workers * seed_concurrency > 4:
        raise PredictionError('combined active seed/prediction concurrency exceeds4')
    if workers == 1:
        return None
    return _trainer().body_hashed({
        'schema': SCHEMA + '.policy', 'prediction_workers_per_seed': workers,
        'seed_concurrency': seed_concurrency,
        'maximum_active_numerical_streams': workers * seed_concurrency,
        'maximum_allowed_active_numerical_streams': 4,
        'native_threads_per_kernel': 1, 'worker_model': 'persistent-spawn',
        'scope': ['common_adjudicator', 'canonical_validation'],
        'model_modes': ['float', 'symmetric-3bit-per-layer-runtime.v1'],
        'channel_quantization_supported': False, 'batch_rows': BATCH_ROWS,
        'assignment': 'original-batch-index-mod-configured-worker-count',
        'model_refresh': 'immutable-current-model-packet-every-call;all-refresh-ACKs-before-prediction',
        'parent_barrier': 'all-helper-done-ACKs-before-parent-metrics-or-training',
        'metrics_reduction': 'unchanged-parent-complete-prediction-vector',
        'feature_guard': 'source-array-content-and-owned-file-hashes-every-call',
        'maximum_feature_bytes': MAX_FEATURE_BYTES,
        'maximum_model_packet_bytes': MAX_MODEL_BYTES,
        'maximum_prediction_calls': MAX_CALLS,
        'automatic_memory_authorization': False,
        'memory_scope': 'separate-source-and-corpus-review-required;CPU-admission-is-not-memory-approval',
        'retry_or_worker_replacement': False,
    })


def validate_policy(value):
    if not isinstance(value, dict) or value != policy(
            value.get('prediction_workers_per_seed'), value.get('seed_concurrency')):
        raise PredictionError('validation prediction policy changed')
    return value


def _record(path):
    path = Path(path).resolve()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}


def _verify(record):
    path = Path(record['path'])
    if path.resolve() != path or _record(path) != record:
        raise PredictionError('prediction source or owned file changed')
    return path


def _array_hash(value):
    return hashlib.sha256(memoryview(value).cast('B')).hexdigest()


def _source_records(trainer):
    directory = Path(__file__).resolve().parent
    if Path(trainer.__file__).resolve() != directory / 'compact_value_bfm_train.py':
        raise PredictionError('prediction helpers cannot mix trainer namespaces')
    records = [_record(path) for path in sorted(directory.glob('*.py'))]
    imported = getattr(trainer, '_TRAINER_SOURCE_AT_IMPORT', None)
    if imported != _record(trainer.__file__)['sha256'] or _SOURCE_AT_IMPORT != _record(__file__)['sha256']:
        raise PredictionError('prediction source changed after module import')
    return records


def _verify_sources(records):
    directory = Path(__file__).resolve().parent
    if {record['path'] for record in records} != {str(p) for p in directory.glob('*.py')}:
        raise PredictionError('prediction source inventory changed')
    for record in records:
        _verify(record)


def _features(dataset, np):
    ptr, idx = dataset.indptr, dataset.indices
    if (ptr.dtype != np.dtype('<i8') or idx.dtype != np.dtype('<u2')
            or ptr.ndim != 1 or idx.ndim != 1 or not ptr.flags.c_contiguous
            or not idx.flags.c_contiguous or len(ptr) < 2 or ptr[0] != 0
            or ptr[-1] != len(idx) or np.any(np.diff(ptr) < 0)):
        raise PredictionError('parallel validation requires canonical contiguous CSR features')
    return {'rows': len(ptr)-1, 'indices': len(idx),
        'indptr_sha256': _array_hash(ptr), 'indices_sha256': _array_hash(idx)}


def model_packet(parameters, architecture, quantized, trainer=None):
    """Serialize primitives and immutable bytes, never live NumPy/class objects."""
    t = trainer or _trainer(); np = t.np
    parameters = t._validate_parameters(parameters, architecture)
    values = []; descriptors = []
    for name in ('w1', 'w2', 'w3'):
        value = parameters[name]
        if not value.flags.c_contiguous:
            raise PredictionError('parallel validation requires C-contiguous model arrays')
        content = value.tobytes(order='C'); values.append(content)
        descriptors.append({'name': name, 'shape': list(value.shape), 'dtype': '<f4'})
    mode = 'float'; scales = None
    if quantized is not None:
        mode = 'runtime.v1'; scales = {}
        if set(quantized.integer) != {'w1', 'w2', 'w3'} or set(quantized.scales) != {'w1', 'w2', 'w3'}:
            raise PredictionError('quantized prediction model roster changed')
        for name, shape in architecture.shapes.items():
            scale = np.asarray(quantized.scales[name])
            if scale.ndim != 0:
                raise PredictionError('parallel validation does not support channel/runtime.v2 quantization')
            number = float(scale)
            value = quantized.integer[name]
            if (not math.isfinite(number) or number <= 0 or float(np.float32(number)) != number
                    or value.dtype != np.dtype('int8') or value.shape != shape
                    or not value.flags.c_contiguous or np.any(value < -3) or np.any(value > 3)):
                raise PredictionError('parallel quantized model is invalid')
            scales[name] = number
            values.append(value.tobytes(order='C'))
            descriptors.append({'name': name, 'shape': list(value.shape), 'dtype': '|i1'})
    header = {'architecture': architecture.name, 'mode': mode, 'scales': scales, 'arrays': descriptors}
    packet = pickle.dumps((header, tuple(values)), protocol=5)
    if len(packet) > MAX_MODEL_BYTES:
        raise PredictionError('prediction model packet exceeds bound')
    return packet, hashlib.sha256(packet).hexdigest(), mode


def _decode_model(packet, digest, t):
    if not isinstance(packet, bytes) or len(packet) > MAX_MODEL_BYTES or hashlib.sha256(packet).hexdigest() != digest:
        raise PredictionError('prediction model packet changed')
    header, buffers = pickle.loads(packet); np = t.np
    if header.get('architecture') not in t.ARCHITECTURES or header.get('mode') not in ('float', 'runtime.v1'):
        raise PredictionError('prediction architecture/model mode changed')
    arch = t.ARCHITECTURES[header['architecture']]
    count = 3 if header['mode'] == 'float' else 6
    if len(buffers) != count or len(header['arrays']) != count:
        raise PredictionError('prediction model array count changed')
    values = []
    for i, (record, data) in enumerate(zip(header['arrays'], buffers, strict=True)):
        name = ('w1', 'w2', 'w3')[i % 3]; dtype = '<f4' if i < 3 else '|i1'
        if record != {'name': name, 'shape': list(arch.shapes[name]), 'dtype': dtype} or type(data) is not bytes:
            raise PredictionError('prediction model array layout changed')
        value = np.frombuffer(data, dtype=dtype).reshape(arch.shapes[name])
        values.append(value)
    parameters = dict(zip(('w1', 'w2', 'w3'), values[:3], strict=True))
    q = None if count == 3 else t.QuantizedWeights(dict(zip(('w1', 'w2', 'w3'), values[3:], strict=True)),
        {name: np.float32(v) for name, v in header['scales'].items()})
    # Re-encode validates scalar scales, finite parameters and forbidden codes.
    if model_packet(parameters, arch, q, t)[0] != packet:
        raise PredictionError('prediction model packet is not canonical')
    return parameters, arch, q


def _usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {'cpu_seconds': usage.ru_utime + usage.ru_stime,
        'peak_rss_bytes': int(usage.ru_maxrss) * (1 if sys.platform == 'darwin' else 1024)}


def _parent_watch(connection):
    try:
        connection.recv_bytes()
    except (EOFError, OSError):
        os._exit(73)
    os._exit(73)


def _worker(slot, spec, commands, replies, parent_watch, parent_pid):
    generation = None
    try:
        if os.getppid() != parent_pid:
            raise PredictionError('prediction parent disappeared before initialization')
        threading.Thread(target=_parent_watch, args=(parent_watch,), daemon=True).start()
        _verify_sources(spec['sources'])
        os.environ.update(ENV); os.environ[MARKER] = '1'
        t = _trainer(); np = t.np
        _source_records(t)
        with t.native_thread_execution_scope() as native:
            maps = {}
            for name, data in spec['datasets'].items():
                ptr = np.load(_verify(data['indptr']), mmap_mode='r', allow_pickle=False)
                idx = np.load(_verify(data['indices']), mmap_mode='r', allow_pickle=False)
                if ptr.flags.writeable or idx.flags.writeable:
                    raise PredictionError('prediction maps are writable')
                maps[name] = (ptr, idx)
            replies.put({'kind': 'ready', 'slot': slot, 'pid': os.getpid(),
                'native_kernel': t.native_kernel_evidence(native),
                'parent_pid': os.getppid(), 'process_group': os.getpgrp(), 'usage': _usage()}, timeout=TIMEOUT_SECONDS)
            parameters = architecture = quantized = None
            while True:
                command = commands.get()
                if command['kind'] == 'close':
                    replies.put({'kind': 'closed', 'slot': slot, 'pid': os.getpid(), 'usage': _usage()}, timeout=TIMEOUT_SECONDS)
                    return
                if command['kind'] != 'refresh' or command['generation'] != (1 if generation is None else generation+1):
                    raise PredictionError('prediction generation is stale or discontinuous')
                generation = command['generation']; digest = command['model_sha256']
                parameters, architecture, quantized = _decode_model(command['model'], digest, t)
                dataset = command['dataset']; ptr, idx = maps[dataset]
                replies.put({'kind': 'refreshed', 'generation': generation, 'model_sha256': digest,
                    'slot': slot, 'pid': os.getpid()}, timeout=TIMEOUT_SECONDS)
                execute = commands.get()
                if execute != {'kind': 'predict', 'generation': generation,
                        'batches': list(range(slot, math.ceil((len(ptr)-1)/BATCH_ROWS), spec['workers']))}:
                    raise PredictionError('prediction assignment changed')
                for ordinal in execute['batches']:
                    start, stop = ordinal*BATCH_ROWS, min((ordinal+1)*BATCH_ROWS, len(ptr)-1)
                    active = tuple(idx[int(ptr[i]):int(ptr[i+1])] for i in range(start, stop))
                    prediction, cache = t.forward(parameters, architecture, active, quantized=quantized)
                    del cache
                    replies.put({'kind': 'batch', 'generation': generation, 'model_sha256': digest,
                        'slot': slot, 'pid': os.getpid(), 'batch': ordinal,
                        'start': start, 'stop': stop, 'prediction': prediction.tobytes()}, timeout=TIMEOUT_SECONDS)
                replies.put({'kind': 'done', 'generation': generation, 'model_sha256': digest,
                    'slot': slot, 'pid': os.getpid(), 'batches': len(execute['batches']), 'usage': _usage()}, timeout=TIMEOUT_SECONDS)
    except BaseException as error:
        try:
            replies.put({'kind': 'error', 'slot': slot, 'pid': os.getpid(), 'generation': generation,
                'error_type': type(error).__name__, 'message': str(error)}, timeout=1)
        except BaseException:
            pass
        raise


class ValidationPredictionPool:
    """One seed's bounded feature maps and persistent, refreshed predictors."""
    def __init__(self, datasets, directory, workers, *, seed_concurrency=1, trainer=None):
        self.t = trainer or _trainer()
        self.policy = policy(workers, seed_concurrency)
        if self.policy is None:
            raise PredictionError('serial prediction does not need a helper pool')
        if set(datasets) != {'common_adjudicator', 'canonical_validation'}:
            raise PredictionError('only scalar common/canonical validation may use prediction helpers')
        self.datasets = dict(datasets); self.directory = Path(directory).resolve()
        self.workers = workers; self.sources = _source_records(self.t)
        self.features = {name: _features(data, self.t.np) for name, data in self.datasets.items()}
        if sum(data.indptr.nbytes + data.indices.nbytes for data in self.datasets.values()) > MAX_FEATURE_BYTES:
            raise PredictionError('validation feature maps exceed the bounded pool capacity')
        self.ctx = multiprocessing.get_context('spawn'); self.processes = []; self.commands = []; self.watches = []
        self.replies = None; self.ready = []; self.calls = []; self.generation = 0; self.closed = False; self.reserved = False; self.close_records = []; self.watch_reads = []; self.start_attempted = []

    def __enter__(self):
        global _RESERVED
        with _BUDGET_LOCK:
            if _RESERVED + self.workers > 4:
                raise PredictionError('process-local helper reservation exceeds4')
            _RESERVED += self.workers; self.reserved = True
        try:
            self.directory.mkdir(parents=True, exist_ok=False)
            self.map_records = {}
            for name, data in self.datasets.items():
                records = {}
                for field in ('indptr', 'indices'):
                    path = self.directory / f'{name}-{field}.npy'
                    with path.open('xb') as handle:
                        self.t.np.save(handle, getattr(data, field), allow_pickle=False)
                    os.chmod(path, 0o444); records[field] = _record(path)
                if _features(data, self.t.np) != self.features[name]:
                    raise PredictionError('validation features changed while snapshotting')
                self.map_records[name] = records
            spec = {'sources': self.sources, 'datasets': self.map_records, 'workers': self.workers}
            self.replies = self.ctx.Queue(maxsize=2*self.workers)
            os.environ.update(ENV); os.environ[MARKER] = '1'
            for slot in range(self.workers):
                commands = self.ctx.Queue(maxsize=1); watch_read, watch_write = self.ctx.Pipe(duplex=False)
                process = self.ctx.Process(target=_worker, args=(slot, spec, commands, self.replies, watch_read, os.getpid()))
                self.processes.append(process); self.commands.append(commands); self.watches.append(watch_write)
                self.watch_reads.append(watch_read); self.start_attempted.append(False)
                self.start_attempted[-1] = True
                process.start()
                watch_read.close()
            self.ready = self._barrier('ready')
            return self
        except BaseException:
            self.close(failed=True)
            raise

    def _receive(self, deadline):
        while True:
            if any(process.exitcode is not None for process in self.processes):
                raise PredictionError('prediction helper exited; no replacement')
            try:
                value = self.replies.get(timeout=min(.1, max(.001, deadline-time.monotonic())))
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise PredictionError('prediction helper response timed out')
                continue
            slot = value.get('slot')
            if type(slot) is not int or not 0 <= slot < self.workers or value.get('pid') != self.processes[slot].pid:
                raise PredictionError('prediction response owner changed')
            if value.get('kind') == 'error':
                raise PredictionError(f"prediction helper failed: {value.get('error_type')}: {value.get('message')}")
            return value

    def _barrier(self, kind, *, model_sha256=None):
        records = {}; deadline = time.monotonic()+TIMEOUT_SECONDS
        while len(records) < self.workers:
            row = self._receive(deadline); slot = row['slot']
            if row.get('kind') != kind or slot in records:
                raise PredictionError('prediction barrier response duplicated or changed')
            if kind != 'ready' and (row.get('generation') != self.generation or row.get('model_sha256') != model_sha256):
                raise PredictionError('stale prediction model acknowledgement')
            records[slot] = row
        return [records[i] for i in range(self.workers)]

    def predict(self, parameters, architecture, dataset, *, quantized=None, batch_size=BATCH_ROWS):
        if self.closed or not self.ready or batch_size != BATCH_ROWS or type(batch_size) is not int:
            raise PredictionError('prediction pool state or original batch size changed')
        name = next((key for key, value in self.datasets.items() if value is dataset), None)
        if name is None:
            raise PredictionError('prediction dataset is outside scalar validation scope')
        if self.generation >= MAX_CALLS:
            raise PredictionError('prediction call evidence exceeded its bound')
        _verify_sources(self.sources)
        if _features(dataset, self.t.np) != self.features[name]:
            raise PredictionError('validation features changed since pool creation')
        for record in self.map_records[name].values():
            _verify(record)
        # Copies are COMPLETE before any asynchronous Queue.put.
        packet, digest, mode = model_packet(parameters, architecture, quantized, self.t)
        self.generation += 1; generation = self.generation
        try:
            for commands in self.commands:
                commands.put({'kind': 'refresh', 'generation': generation, 'dataset': name,
                    'model': packet, 'model_sha256': digest}, timeout=TIMEOUT_SECONDS)
            self._barrier('refreshed', model_sha256=digest)
            count = math.ceil(len(dataset)/BATCH_ROWS)
            for slot, commands in enumerate(self.commands):
                commands.put({'kind': 'predict', 'generation': generation,
                    'batches': list(range(slot, count, self.workers))}, timeout=TIMEOUT_SECONDS)
            np = self.t.np; output = np.empty(len(dataset), dtype=np.float32)
            seen = set(); done = {}; deadline = time.monotonic()+TIMEOUT_SECONDS
            while len(seen) < count or len(done) < self.workers:
                row = self._receive(deadline); slot = row['slot']
                if row.get('generation') != generation or row.get('model_sha256') != digest:
                    raise PredictionError('prediction response uses a stale model')
                if row['kind'] == 'done':
                    if slot in done or row.get('batches') != len(range(slot,count,self.workers)):
                        raise PredictionError('prediction completion count changed')
                    done[slot] = row
                    continue
                ordinal = row.get('batch')
                if row['kind'] != 'batch' or type(ordinal) is not int or ordinal not in range(count) or ordinal in seen or ordinal % self.workers != slot or slot in done:
                    raise PredictionError('prediction batch order/identity changed')
                start, stop = ordinal*BATCH_ROWS, min((ordinal+1)*BATCH_ROWS,len(dataset))
                data = row.get('prediction')
                if type(data) is not bytes or len(data) != 4*(stop-start) or (row['start'],row['stop']) != (start,stop):
                    raise PredictionError('prediction batch shape changed')
                values = np.frombuffer(data,dtype=np.float32)
                if not np.all(np.isfinite(values)):
                    raise PredictionError('prediction result is nonfinite')
                output[start:stop] = values; seen.add(ordinal)
            self.calls.append({'generation': generation, 'dataset': name, 'model_mode': mode,
                'model_sha256': digest, 'rows':len(dataset),'batches':count,
                'active_slots':list(range(min(count,self.workers))),
                'all_helpers_refreshed_and_done':True,'prediction_sha256':_array_hash(output),
                'helper_usage': [done[i]['usage'] for i in range(self.workers)]})
            return output
        except BaseException:
            self.close(failed=True)
            raise

    def close(self, *, failed=False):
        global _RESERVED
        if self.closed:
            return
        errors = []
        try:
            if not failed and self.ready:
                for commands in self.commands:
                    commands.put({'kind':'close'},timeout=TIMEOUT_SECONDS)
                records = {}; deadline=time.monotonic()+TIMEOUT_SECONDS
                while len(records)<len(self.processes):
                    # A closed acknowledgement may precede normal process exit.
                    try: row=self.replies.get(timeout=.1)
                    except queue.Empty:
                        if time.monotonic()>deadline:raise PredictionError('prediction close timed out')
                        continue
                    slot=row.get('slot')
                    if row.get('kind')!='closed' or type(slot) is not int or slot in records or not 0<=slot<len(self.processes) or row.get('pid')!=self.processes[slot].pid:
                        raise PredictionError('prediction close acknowledgement changed')
                    records[slot]=row
                self.close_records = [records[i] for i in range(self.workers)]
        except BaseException as error:
            failed=True;errors.append(error)
        finally:
            for process in self.processes:
                if failed and process.pid is not None and process.is_alive():process.terminate()
            for process in self.processes:
                if process.pid is None:continue
                process.join(timeout=5)
                if process.is_alive():process.kill();process.join(timeout=5)
                if process.is_alive():errors.append(PredictionError('prediction helper could not be joined'))
                elif not failed and process.exitcode!=0:errors.append(PredictionError('prediction helper closed unsuccessfully'))
            for watch in (*self.watches, *self.watch_reads):
                try:watch.close()
                except OSError:pass
            for commands in (*self.commands, *([] if self.replies is None else [self.replies])):
                commands.close();commands.cancel_join_thread()
            alive = any(process.pid is not None and process.is_alive() for process in self.processes)
            unknown = any(attempted and process.pid is None for attempted,process in zip(self.start_attempted,self.processes))
            alive = alive or unknown
            if self.reserved and not alive:
                with _BUDGET_LOCK:_RESERVED-=self.workers
                self.reserved=False
            self.closed=not alive;self.successful_close=not failed and not errors and not alive
            if alive:
                raise PredictionError('prediction helper ownership unresolved; reservation retained')
        if errors and not failed:
            raise errors[0]
        if errors and self.ready and not self.successful_close:
            raise PredictionError('prediction helper cleanup failed') from errors[0]

    def evidence(self):
        if not self.closed or not self.successful_close:
            raise PredictionError('prediction execution has no successful owned closure')
        return self.t.body_hashed({'schema':SCHEMA+'.execution','policy':self.policy,
            'sources':self.sources,'datasets':{name:{'features':self.features[name],**self.map_records[name]} for name in self.datasets},
            'helpers':self.ready,'calls':self.calls,'closed':True,'close_records':self.close_records,
            'coordinator_pid':os.getpid(),'coordinator_usage':_usage(),
            'all_helper_exitcodes':[p.exitcode for p in self.processes],
            'raw_predictions_retained':False,'model_packets_retained':False})

    def __exit__(self, kind, value, traceback):
        self.close(failed=kind is not None)
        return False


def validate_execution(value, settings, *, artifact_root, dataset_identities, trainer=None):
    """Check optional provenance/owned files without replaying predictions."""
    t = trainer or _trainer(); np = t.np
    if not isinstance(value, dict):
        raise PredictionError('validation prediction execution is absent')
    t.verify_body_hash(value, schema=SCHEMA+'.execution', label='validation prediction execution')
    expected_fields={'schema','policy','sources','datasets','helpers','calls','closed','close_records',
        'coordinator_pid','coordinator_usage','all_helper_exitcodes','raw_predictions_retained','model_packets_retained','body_sha256'}
    if set(value)!=expected_fields or value['policy']!=settings['policy'] or value['sources']!=settings['sources']:
        raise PredictionError('validation prediction execution policy/source changed')
    pol=validate_policy(value['policy']);workers=pol['prediction_workers_per_seed']
    root=Path(artifact_root).resolve()
    if set(value['datasets'])!={'common_adjudicator','canonical_validation'}:
        raise PredictionError('validation predictor dataset scope changed')
    if {name:entry.get('features') for name,entry in value['datasets'].items()}!=settings.get('features'):
        raise PredictionError('validation prediction features differ from outer binding')
    # Preflight every owned path before opening any feature file.
    for entry in value['datasets'].values():
        for key in ('indptr','indices'):
            path=Path(entry[key]['path'])
            if path.resolve()!=path or not path.is_relative_to(root):
                raise PredictionError('prediction feature artifact escaped output root')
    for record in value['sources']:
        if Path(record['path']).suffix!='.py':raise PredictionError('prediction source record is not Python')
        _verify(record)
    total=0
    for name,entry in value['datasets'].items():
        ptr=np.load(_verify(entry['indptr']),mmap_mode='r',allow_pickle=False)
        idx=np.load(_verify(entry['indices']),mmap_mode='r',allow_pickle=False)
        features=_features(type('FeatureView',(),{'indptr':ptr,'indices':idx})(),np)
        if features!=entry['features'] or features['rows']!=dataset_identities[name]['samples'] or features['indices']!=dataset_identities[name]['active_features']:
            raise PredictionError('validation prediction feature identity differs')
        total+=ptr.nbytes+idx.nbytes
    if total>MAX_FEATURE_BYTES or value['closed'] is not True or value['all_helper_exitcodes']!=[0]*workers or value['raw_predictions_retained'] is not False or value['model_packets_retained'] is not False:
        raise PredictionError('validation prediction closure/resource evidence changed')
    helpers=value['helpers'];closed=value['close_records']
    if len(helpers)!=workers or len(closed)!=workers or type(value['coordinator_pid']) is not int or value['coordinator_pid']<=0:
        raise PredictionError('prediction owner roster is incomplete')
    pids=[]
    def usage(u):
        return (isinstance(u,dict) and set(u)=={'cpu_seconds','peak_rss_bytes'}
            and type(u['cpu_seconds']) in (int,float) and math.isfinite(u['cpu_seconds']) and u['cpu_seconds']>=0
            and type(u['peak_rss_bytes']) is int and u['peak_rss_bytes']>0)
    if not usage(value['coordinator_usage']):raise PredictionError('coordinator usage is invalid')
    for i,helper in enumerate(helpers):
        if helper.get('kind')!='ready' or helper.get('slot')!=i or type(helper.get('pid')) is not int or helper['pid']<=0 or helper.get('parent_pid')!=value['coordinator_pid'] or not usage(helper.get('usage')):
            raise PredictionError('prediction helper ownership changed')
        t.validate_native_kernel_evidence(helper['native_kernel']);pids.append(helper['pid'])
        if closed[i].get('kind')!='closed' or closed[i].get('slot')!=i or closed[i].get('pid')!=helper['pid'] or not usage(closed[i].get('usage')):
            raise PredictionError('prediction helper closure changed')
    if len(set(pids))!=workers or value['coordinator_pid'] in pids:
        raise PredictionError('prediction helper PIDs are not distinct')
    calls=value['calls']
    if not isinstance(calls,list) or not 1<=len(calls)<=MAX_CALLS:
        raise PredictionError('prediction call roster is absent or over bound')
    for generation,call in enumerate(calls,start=1):
        name=call.get('dataset')
        if name not in value['datasets']:raise PredictionError('prediction call escaped scalar validation')
        rows=value['datasets'][name]['features']['rows'];count=math.ceil(rows/BATCH_ROWS)
        if (type(call.get('generation')) is not int or call['generation']!=generation
                or call.get('model_mode') not in ('float','runtime.v1') or not t.valid_sha256(call.get('model_sha256'))
                or call.get('rows')!=rows or call.get('batches')!=count
                or call.get('active_slots')!=list(range(min(count,workers)))
                or call.get('all_helpers_refreshed_and_done') is not True or not t.valid_sha256(call.get('prediction_sha256'))
                or len(call.get('helper_usage',[]))!=workers or not all(usage(u) for u in call['helper_usage'])):
            raise PredictionError('prediction generation/model/batch evidence changed')
    return value
