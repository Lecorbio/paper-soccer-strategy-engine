"""Retain settled nonactivation attempts and guard one fresh UI retry."""
from __future__ import annotations

from pathlib import Path

from tools import rank_two_live_eight_v1 as c
from tools import rank_two_live_eight_windows_v1 as retained


def identity(snapshot):
    user = snapshot['user']
    submissions = {p['submissionId'] for game in snapshot['battles'] for p in game['players']
                   if p.get('playerAgentId') == user['agentId']}
    if user.get('percentage') != 100 or len(submissions) != 1 or not all(g.get('done') for g in snapshot['battles']):
        raise PermissionError('a complete singular predecessor identity is required')
    return user['agentId'], next(iter(submissions)), user.get('testSessionHandle')


def unchanged(before, after):
    if identity(before) != identity(after):
        raise PermissionError('activation changed; attest the original claim before any retry')
    if {g['gameId'] for g in before['battles']} != {g['gameId'] for g in after['battles']}:
        raise PermissionError('predecessor games changed; archive before retry')


def claim_retry(base, aid, usage, copyback):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    declared = state['live_eight_declared_windows'][aid]
    sha = declared['source_sha256']
    c.guard(base, 'claim', state['owner_thread_id'], usage, source=sha, attempt=aid)
    resolution_ref = state['live_eight_claim_resolutions'][aid]
    resolution = c.read(c.verify(resolution_ref))
    if (resolution.get('passed') is not True or resolution.get('kind') != 'settled-without-new-activation'
            or resolution.get('source_sha256') != sha or resolution.get('retry_actions_authorized') != 1
            or resolution.get('new_identity_attested') is not False):
        raise PermissionError('bound nonactivation resolution and single retry required')
    root = base / 'live-eight-v1'
    m = retained.adapter(root)
    with m.lock(root):
        out, plan = m.attempt(root, aid)
        m.check_lease(out)
        pending = m.next_slot(root)
        if pending is None or pending[1]['attempt_id'] != aid:
            raise PermissionError('retry is not the current predeclared slot')
        if resolution['original_claim'] != c.record(out / 'claim.json'):
            raise PermissionError('original claim binding differs')
        retry = out / 'retries' / '01'
        if (retry / 'claim.json').exists():
            raise PermissionError('retry already claimed; observe only')
        cp = Path(copyback).resolve()
        if cp == Path(plan['source']['path']).resolve() or c.record(cp)['sha256'] != sha:
            raise PermissionError('fresh independent retry copyback differs')
        capture = c.immutable(retry / 'editor-copyback.cpp', cp.read_bytes())
        before = m.snapshot(retry / 'before')
        unchanged(c.read(c.verify(resolution['latest_server_snapshot'])), before)
        claim = c.immutable(retry / 'claim.json', dict(plan=m.e.raw_record(out / 'plan.json'),
            copyback=capture, before=c.record(retry / 'before/snapshot.json'), resolution=resolution_ref,
            claimed_at=c.now().isoformat(), clicks_permitted=1, retry_click_allowed=False,
            producer=c.record(__file__)))
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        if current['active_claims']:
            raise PermissionError('another unresolved claim appeared')
        current['active_claims'][aid] = dict(source_sha256=sha, claim=claim)
        current.setdefault('live_eight_effective_claims', {})[aid] = claim
        c.atomic(base / 'CURRENT.json', current)
    return claim


def intent_retry(base, aid, usage):
    base = Path(base)
    state = c.read(base / 'CURRENT.json')
    sha = state['live_eight_declared_windows'][aid]['source_sha256']
    c.guard(base, 'submit', state['owner_thread_id'], usage, source=sha, attempt=aid)
    claim = state['live_eight_effective_claims'][aid]
    c.verify(claim)
    return c.immutable(Path(claim['path']).parent / 'UI_SUBMIT_INTENT.json', dict(
        claim=claim, source_sha256=sha, exactly_one_action=True, created_at=c.now().isoformat()))


def attest_retry(base, aid, usage):
    base = Path(base)
    root = base / 'live-eight-v1'
    state = c.read(base / 'CURRENT.json')
    sha = state['live_eight_declared_windows'][aid]['source_sha256']
    c.guard(base, 'observe', state['owner_thread_id'], usage, source=sha, attempt=aid)
    m = retained.adapter(root)
    with m.lock(root):
        out, plan = m.attempt(root, aid)
        if (out / 'submission.json').exists():
            return c.read(out / 'submission.json')
        claim_ref = state['live_eight_effective_claims'][aid]
        claim = c.read(c.verify(claim_ref))
        if claim['plan'] != m.e.raw_record(out / 'plan.json') or c.record(c.verify(claim['copyback']))['sha256'] != sha:
            raise PermissionError('retry claim or copyback changed')
        before = c.read(c.verify(claim['before']))
        retry = Path(claim_ref['path']).parent
        number = len(list((retry / 'identity').glob('*/snapshot.json')))
        directory = retry / 'identity' / f'{number:04d}'
        after = m.snapshot(directory)
        ids = m.submission_identity(before, after)
        value = dict(**ids, identity_protocol='fresh-submission-owner-session-v1',
            identity_producer=c.record(__file__), source_sha256=sha, attempt_id=aid,
            claim=claim_ref, attested_at=c.now().isoformat(), observation=c.record(directory / 'snapshot.json'))
        reference = c.immutable(out / 'submission.json', value)
        m.atomic(root / 'active.json', value)
    with c.locked(base):
        current = c.read(base / 'CURRENT.json')
        current['live_eight_declared_windows'][aid]['attested'] = reference
        current['live_eight_known_windows'][aid] = dict(source_sha256=sha, submission=reference, all90_archived=False)
        current['active_claims'].pop(aid, None)
        current['live_eight_active_attempt'] = aid
        c.atomic(base / 'CURRENT.json', current)
    return value
