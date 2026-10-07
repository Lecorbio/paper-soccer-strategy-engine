#!/usr/bin/env python3
"""Source-bound rank-two queue with one shared, bounded workload pool.

The git-common-dir lock excludes every v3 queue in this repository's worktrees.
Older queue versions and unrelated workloads must be idle during this campaign,
especially authoritative timing. Limits are sampled (~250 ms plus ps overhead),
not kernel containment. Trusted commands must preserve the inherited ownership
tag and keep acceleration disabled. All workload threads count, including idle
launchers; at most one explicit coordinator thread is excluded from max_workers. A hash-bound
native referee may declare its two process-supervisor threads separately.
max_workers bounds resident thread capacity, not concurrently searching engines.
One authoritative match may keep both native bot processes resident.
RSS includes the queue supervisor plus a 64 MiB reserve for transient ps helpers.
CPU capacity is elapsed wall time times declared workload threads plus supervisor,
not measured consumed CPU. TERM allows checkpoints before the bounded KILL grace.
"""
from __future__ import annotations
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import math
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'papersoccer.experiment-queue.v4'

def record(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

def verify(ref):
    if record(ref['path']) != ref:
        raise ValueError('changed input: ' + ref['path'])

def read(path):
    return json.loads(Path(path).read_text())

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)

@contextlib.contextmanager
def lock(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / 'supervisor.lock').open('a+') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('queue already has an active supervisor')
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)

def identity(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], text=True, capture_output=True)
    return ' '.join(result.stdout.split()) or None

def live(owner):
    return owner and owner.get('birth') is not None and identity(owner['pid']) == owner['birth']

def validate(spec):
    if spec.get('schema') != SCHEMA or not re.fullmatch('[a-z0-9][a-z0-9_-]{0,63}', spec['id']):
        raise ValueError('invalid schema/id')
    if Path(spec['cwd']).resolve() != ROOT: raise ValueError('job must use existing worktree')
    if not spec.get('argv') or any(not isinstance(x, str) for x in spec['argv']): raise ValueError('argv required')
    limits = spec['limits']
    if not 1 <= limits['max_processes'] <= 11 or not 1 <= limits['max_workers'] <= 10 or not 0 < limits['rss_bytes'] <= 18*1024**3:
        raise ValueError('resource limits exceed campaign allowance')
    if any(not math.isfinite(limits[k]) or limits[k] <= 0 for k in ('wall_seconds', 'cpu_capacity_seconds', 'termination_grace_seconds')):
        raise ValueError('finite positive budgets required')
    if limits.get('coordinator_threads') not in (0, 1): raise ValueError('at most one coordinator thread')
    if any(type(limits[k]) is not int for k in ('max_workers', 'max_processes', 'coordinator_threads')):
        raise ValueError('integer process and thread allowances required')
    if limits['termination_grace_seconds'] > 30: raise ValueError('termination grace exceeds 30 seconds')
    if spec.get('acceleration') != 'disabled': raise ValueError('acceleration has no explicit memory accounting')
    if spec.get('workload') not in ('fixed-work', 'authoritative-timing'): raise ValueError('explicit workload required')
    auxiliary = limits.get('native_supervisor_threads', 0)
    if auxiliary not in (0, 2) or type(auxiliary) is not int:
        raise ValueError('native referee has exactly two auxiliary supervisors')
    if auxiliary:
        referee = spec.get('native_referee')
        if spec['workload'] != 'authoritative-timing' or not referee or Path(spec['argv'][0]).resolve() != Path(referee['path']).resolve():
            raise ValueError('auxiliary threads require a bound native referee command')
        verify(referee)
    if spec['workload'] == 'authoritative-timing' and (limits['max_workers'] > 2 or limits['max_processes'] > 3 + auxiliary or spec.get('matches') != 1):
        raise ValueError('authoritative timing requires one match and at most two resident bot threads')
    if not spec.get('inputs') or not spec.get('compiler') or not spec.get('acceptance'):
        raise ValueError('immutable inputs, compiler and acceptance required')
    for ref in [*spec['inputs'], spec['compiler']['executable']]: verify(ref)
    if subprocess.check_output([spec['compiler']['executable']['path'], '--version'], text=True) != spec['compiler']['version']:
        raise ValueError('compiler identity changed')
    if spec.get('resume_policy') not in ('never', 'transactional'):
        raise ValueError('explicit resume policy required')
    for condition in spec['acceptance']:
        if set(condition) != {'path', 'pointer', 'equals'}: raise ValueError('acceptance requires exact JSON predicate')

def enqueue(root, spec):
    validate(spec)
    directory = Path(root) / 'jobs' / spec['id']
    with lock(root):
        path = directory / 'job.json'
        if path.exists():
            if read(path) != spec: raise ValueError('immutable job identity changed')
        else:
            atomic(path, spec); atomic(directory / 'state.json', {'status': 'pending', 'attempts': 0, 'job': record(path)})
    return status(root)

def status(root):
    rows = []
    for path in sorted((Path(root) / 'jobs').glob('*/state.json')):
        state = read(path)
        rows.append({'id': path.parent.name, **state, 'owner_alive': live(state.get('owner'))})
    return {'schema': SCHEMA, 'jobs': rows}

OWNED = {}

TOKENS = {}

def shared_root():
    """All v3 queues in every worktree share this lock; older queues must be idle."""
    common = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=ROOT, text=True).strip()
    return (ROOT / common).resolve() / 'rank-two-resources'

def memory_pressure():
    # macOS uses 1=normal, 2=warning, 4=critical. Unreadable pressure fails closed.
    if sys.platform != 'darwin': return 'normal'
    try:
        value = subprocess.check_output(['sysctl', '-n', 'kern.memorystatus_vm_pressure_level'], text=True).strip()
        return 'normal' if value == '1' else 'pressure'
    except (OSError, subprocess.CalledProcessError): return 'unknown'

def process_table():
    fields = 'pid=,ppid=,rss=,stat=,lstart='
    threads = {}
    if sys.platform == 'darwin':
        for line in subprocess.check_output(['ps', '-M', '-axo', 'pid='], text=True).splitlines():
            parts = line.split()
            value = parts[0] if parts and parts[0].isdigit() else (parts[1] if len(parts) > 1 else '')
            if value.isdigit():
                pid = int(value); threads[pid] = threads.get(pid, 0) + 1
    else:
        for line in subprocess.check_output(['ps', '-axo', 'pid=,nlwp='], text=True).splitlines():
            pid, count = line.split(); threads[int(pid)] = int(count)
    output = subprocess.check_output(['ps', '-axo', fields], text=True)
    rows = {}
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and not parts[3].startswith('Z'):
            pid, parent, rss, _, birth = parts; pid = int(pid)
            # Newly spawned processes may be absent from the first snapshot.
            if pid not in threads and sys.platform == 'darwin':
                detail = subprocess.run(['ps', '-M', '-p', str(pid)], text=True, capture_output=True)
                count = sum(1 for line in detail.stdout.splitlines()[1:] if line.strip())
                if count == 0: continue
                threads[pid] = count
            rows[pid] = {'pid': pid, 'parent': int(parent), 'rss': int(rss)*1024,
                         'threads': threads.get(pid, 1000000), 'birth': ' '.join(birth.split())}
    return rows

def tagged_processes(token):
    # Inherited tag also finds double-forked / new-session children between polls.
    # Commands are trusted: deliberately stripping this environment is unsupported.
    output = subprocess.check_output(['ps', 'eww', '-axo', 'pid=,command='], text=True)
    marker = 'PAPER_SOCCER_QUEUE_TOKEN=' + token
    return {int(line.split(None, 1)[0]) for line in output.splitlines() if marker in line}

def owned_processes(root_pid):
    table = process_table()
    known = OWNED.setdefault(root_pid, {})
    if not known and root_pid in table: known[root_pid] = table[root_pid]['birth']
    current = {pid for pid, birth in known.items() if pid in table and table[pid]['birth'] == birth}
    if root_pid in TOKENS:
        current.update(tagged_processes(TOKENS[root_pid]) & table.keys())
        known.update({pid: table[pid]['birth'] for pid in current})
    while True:
        added = {pid for pid, row in table.items() if row['parent'] in current} - current
        if not added: break
        current.update(added)
        known.update({pid: table[pid]['birth'] for pid in added})
    return [table[pid] for pid in sorted(current)]

def group_usage(root_pid):
    rows = owned_processes(root_pid)
    return len(rows), sum(row['rss'] for row in rows)

def stop_group(root_pid, grace=2):
    # Referee bot supervisors intentionally create new sessions. Follow parentage
    # and remembered births, rather than assuming a shared process group.
    if not owned_processes(root_pid): return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for row in reversed(owned_processes(root_pid)):
            if identity(row['pid']) != row['birth']: continue
            try: os.kill(row['pid'], sig)
            except ProcessLookupError: pass
        if sig == signal.SIGTERM: time.sleep(grace)

def acceptable(spec):
    for condition in spec['acceptance']:
        value = read(condition['path'])
        for part in condition['pointer'].strip('/').split('/') if condition['pointer'].strip('/') else []:
            value = value[int(part)] if isinstance(value, list) else value[part]
        if value != condition['equals']: return False
    return True

def _run(root, max_jobs=1, pressure=memory_pressure):
    root = Path(root)
    with lock(root):
        finished = 0
        for row in status(root)['jobs']:
            if finished >= max_jobs: break
            if row['status'] != 'pending': continue
            if pressure() != 'normal': break
            directory = root / 'jobs' / row['id']; spec = read(directory / 'job.json')
            verify(row['job']); validate(spec)
            states = {v['id']: v['status'] for v in status(root)['jobs']}
            if any(states.get(dep) != 'complete' for dep in spec.get('dependencies', [])): continue
            number = row['attempts'] + 1; attempt = directory / f'attempt-{number:03d}'
            attempt.mkdir()
            owner = {'pid': os.getpid(), 'birth': identity(os.getpid())}
            state = {'status': 'running', 'attempts': number, 'job': row['job'], 'owner': owner}
            atomic(directory / 'state.json', state)
            started = time.monotonic(); limits = spec['limits']; peak_rss = peak_processes = peak_threads = 0
            reason = None; proc = None; code = None
            try:
                with (attempt / 'console.log').open('xb') as log:
                    token = uuid.uuid4().hex
                    env = dict(os.environ, PAPER_SOCCER_QUEUE_TOKEN=token, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
                    proc = subprocess.Popen(spec['argv'], cwd=spec['cwd'], stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True, close_fds=True, env=env)
                    TOKENS[proc.pid] = token
                    atomic(attempt / 'claim.json', {'job': row['job'], 'owner': owner,
                           'child': {'pid': proc.pid, 'birth': identity(proc.pid)}, 'started_unix': time.time(), 'token': token})
                    while proc.poll() is None:
                        rows = owned_processes(proc.pid)
                        count = len(rows); threads = sum(v['threads'] for v in rows)
                        # Include the queue supervisor in the aggregate resident-memory ceiling.
                        rss = sum(v['rss'] for v in rows) + process_table()[os.getpid()]['rss'] + 64*1024**2
                        peak_threads = max(peak_threads, threads)
                        peak_rss = max(peak_rss, rss); peak_processes = max(peak_processes, count)
                        ownership = [{'pid': pid, 'birth': birth} for pid, birth in OWNED.get(proc.pid, {}).items()]
                        ownership_path = attempt / 'ownership.json'
                        if not ownership_path.exists() or read(ownership_path) != ownership: atomic(ownership_path, ownership)
                        elapsed = time.monotonic() - started
                        if rss > limits['rss_bytes']: reason = 'rss-limit'
                        elif count > limits['max_processes']: reason = 'process-limit'
                        elif threads > limits['max_workers'] + limits['coordinator_threads'] + limits.get('native_supervisor_threads', 0): reason = 'worker-thread-limit'
                        elif pressure() != 'normal': reason = 'memory-pressure'
                        elif elapsed > limits['wall_seconds']: reason = 'wall-limit'
                        # Conservative CPU upper bound; includes sleeping process capacity.
                        elif elapsed * (limits['max_workers'] + limits['coordinator_threads'] + limits.get('native_supervisor_threads', 0) + 1) > limits['cpu_capacity_seconds']: reason = 'cpu-capacity-limit'
                        if reason: break
                        time.sleep(.25)
                    if reason: stop_group(proc.pid, limits['termination_grace_seconds'])
                    code = proc.wait()
                if not reason and owned_processes(proc.pid): reason = 'descendants-outlived-command'
                if not reason and code != 0: reason = 'nonzero-exit'
                for ref in [*spec['inputs'], spec['compiler']['executable']]: verify(ref)
                if not reason and not acceptable(spec): reason = 'acceptance-failed'
            except BaseException as exc:
                reason = type(exc).__name__ + ': ' + str(exc)
            finally:
                if proc is not None:
                    stop_group(proc.pid, limits['termination_grace_seconds']); proc.wait()
                    TOKENS.pop(proc.pid, None)
            packet = {'job': row['job'], 'attempt': number, 'status': 'complete' if reason is None else 'stopped',
                      'reason': reason, 'exit_code': code, 'wall_seconds': time.monotonic()-started,
                      'peak_rss_bytes': peak_rss, 'peak_processes': peak_processes, 'peak_threads': peak_threads, 'native_supervisor_threads': limits.get('native_supervisor_threads', 0),
                      'cpu_charge_method': 'elapsed wall seconds times worker + coordinator + supervisor thread capacity',
                      'next_decision': 'eligible dependent jobs' if reason is None else 'classify attempt before explicit resume'}
            packet['cpu_capacity_seconds'] = packet['wall_seconds'] * (limits['max_workers'] + limits['coordinator_threads'] + limits.get('native_supervisor_threads', 0) + 1)
            atomic(attempt / 'result.json', packet)
            state.update(status=packet['status'], result=record(attempt / 'result.json'))
            atomic(directory / 'state.json', state); finished += 1
            if reason: break
    return status(root)

def run(root, max_jobs=1, pressure=memory_pressure):
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise InterruptedError('supervisor received SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        with lock(shared_root()):
            lease = shared_root() / 'last-queue.json'
            if lease.exists():
                previous_queue = Path(read(lease)['queue'])
                if not previous_queue.exists():
                    return {**status(root), 'admission': 'recovery-required', 'reason': 'prior queue unavailable'}
                for job in status(previous_queue)['jobs']:
                    if job['status'] == 'running':
                        return {**status(root), 'admission': 'recovery-required', 'reason': 'prior running claim requires explicit recovery'}
                    claim = previous_queue / 'jobs' / job['id'] / f"attempt-{job['attempts']:03d}" / 'claim.json'
                    if claim.exists() and tagged_processes(read(claim).get('token', 'missing')):
                        return {**status(root), 'admission': 'recovery-required', 'reason': 'prior descendants still active'}
            atomic(lease, {'queue': str(Path(root).resolve())})
            result = _run(root, max_jobs, pressure)
            if all(job['status'] != 'running' for job in result['jobs']): lease.unlink()
            return result
    except RuntimeError as exc:
        if 'active supervisor' not in str(exc): raise
        return {**status(root), 'admission': 'busy', 'reason': 'shared campaign resource lock held'}
    finally:
        signal.signal(signal.SIGTERM, previous)

def resume(root, job_id, classification):
    if not classification.strip(): raise ValueError('interrupted attempt classification required')
    with lock(root):
        directory = Path(root) / 'jobs' / job_id
        state = read(directory / 'state.json'); verify(state['job']); spec = read(directory / 'job.json'); validate(spec)
        if state['status'] not in ('running', 'stopped'): raise ValueError('job is not interrupted/stopped')
        if live(state.get('owner')): raise ValueError('owner still active')
        claim = directory / f"attempt-{state['attempts']:03d}" / 'claim.json'
        if not claim.exists(): raise ValueError('unknown claim; create an explicit successor, never replay')
        if not read(claim).get('token'): raise ValueError('unknown claim token; never replay')
        if tagged_processes(read(claim)['token']):
            raise ValueError('tagged descendant still active; no replay')
        if claim.exists() and live(read(claim).get('child')): raise ValueError('orphan child still active; classify and stop explicitly')
        ownership = directory / f"attempt-{state['attempts']:03d}" / 'ownership.json'
        if ownership.exists() and any(live(owner) for owner in read(ownership)):
            raise ValueError('owned descendant still active; no duplicate supervisor')
        if spec['resume_policy'] != 'transactional': raise ValueError('spent work requires explicit successor job, never replay')
        atomic(directory / f"recovery-{state['attempts']:03d}.json", {'prior': state, 'classification': classification})
        state.update(status='pending', owner=None); atomic(directory / 'state.json', state)
    return status(root)

def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--queue', type=Path, required=True)
    s = p.add_subparsers(dest='command', required=True)
    q = s.add_parser('enqueue'); q.add_argument('--spec', type=Path, required=True)
    q = s.add_parser('run'); q.add_argument('--max-jobs', type=int, default=1)
    q = s.add_parser('resume'); q.add_argument('--job', required=True); q.add_argument('--classification', required=True)
    s.add_parser('status'); s.add_parser('report'); a = p.parse_args()
    if a.command == 'enqueue': result = enqueue(a.queue, read(a.spec))
    elif a.command == 'run':
        if not 1 <= a.max_jobs <= 100: p.error('max-jobs must be 1..100')
        result = run(a.queue, a.max_jobs)
    elif a.command == 'resume': result = resume(a.queue, a.job, a.classification)
    else: result = status(a.queue)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
