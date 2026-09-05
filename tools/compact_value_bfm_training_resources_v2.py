#!/usr/bin/env python3
"""Bind an explicit four-process resource exception without changing old policy.

Existing phase receipts retain their original two-worker limit. A newly frozen
phase may select four spawned seed workers using this separate user authority.
The one-time concurrent equivalence experiment has a distinct temporary 2+4 cap;
its contended timing does not establish a speed improvement.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign

SCHEMA = campaign.ID + '.training-resource-authorization.v2'
USER_REQUEST = 'can you do this on more cores to make it faster'
TRAINING = {'executor': 'spawn-v2', 'maximum_seed_workers': 4,
            'native_threads_per_worker': 1, 'future_phases_only': True}
BENCHMARK = {'existing_workers': 2, 'benchmark_workers': 4, 'temporary_total_seed_workers': 6,
             'native_threads_per_worker': 1, 'timing_is_speed_evidence': False,
             'current_training_restart_authorized': False}


def authorization_path(root):
    return Path(root).resolve() / 'training-resources/more-cores-authorization.json'


def _body(root, *, authorized_at_utc, user_request, producer):
    root = Path(root).resolve()
    if not isinstance(authorized_at_utc, str):
        raise ValueError('resource authorization requires a truthful UTC timestamp')
    try:
        parsed = dt.datetime.fromisoformat(authorized_at_utc.replace('Z', '+00:00'))
    except ValueError as error:
        raise ValueError('invalid resource authorization timestamp') from error
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError('resource authorization timestamp must identify UTC')
    if user_request != USER_REQUEST:
        raise ValueError('resource authorization must quote the explicit more-cores request')
    parent = campaign.read(root / 'campaign.json')
    if parent.get('policy') != campaign.POLICY or parent['policy']['workers']['training_seeds_max'] != 2:
        raise ValueError('the historical two-worker campaign policy must remain unchanged')
    return {'schema': SCHEMA, 'campaign': campaign.record(root / 'campaign.json'),
            'authorized_by': 'user', 'user_request': user_request, 'authorized_at_utc': authorized_at_utc,
            'training': TRAINING, 'benchmark': BENCHMARK,
            'original_campaign_policy_modified': False, 'existing_phase_receipts_modified': False,
            'data_architecture_training_math_and_gates_unchanged': True,
            'source_bound_equivalence_required_before_production_activation': True,
            'new_training_started': False, 'campaign_success': False, 'producer': producer}


def authorize(root, *, authorized_at_utc, user_request):
    """Record already supplied user authority; never infer it from task activity."""
    root = Path(root).resolve()
    path = authorization_path(root)
    if path.exists():
        existing = validate_authorization(campaign.record(path), root)
        if existing['authorized_at_utc'] != authorized_at_utc or existing['user_request'] != user_request:
            raise ValueError('existing resource authorization cannot be replaced')
        return existing
    # Validate before writing even an auxiliary producer copy.
    _body(root, authorized_at_utc=authorized_at_utc, user_request=user_request,
          producer=campaign.record(Path(__file__)))
    producer = campaign.copy_checked(Path(__file__), path.parent / 'sources' / (campaign.sha(Path(__file__)) + '.py'))
    return campaign.seal(path, _body(root, authorized_at_utc=authorized_at_utc, user_request=user_request, producer=producer))


def validate_authorization(record, root):
    root = Path(root).resolve()
    expected = authorization_path(root)
    if (not isinstance(record, dict) or Path(record.get('path', '')).absolute() != expected
            or expected.resolve() != expected):
        raise ValueError('resource authorization is outside its fixed campaign path')
    document = campaign.read(campaign.verify(record))
    producer = document['producer']
    source = campaign.verify(producer)
    if source.resolve().parent != expected.parent / 'sources' or source.name != producer['sha256'] + '.py':
        raise ValueError('resource authorization lost its immutable producer binding')
    body = _body(root, authorized_at_utc=document['authorized_at_utc'], user_request=document['user_request'], producer=producer)
    if {key: value for key, value in document.items() if key != 'body_sha256'} != body:
        raise ValueError('resource authorization changed its four-worker scope or frozen campaign')
    return document


def expected_workers(contract):
    setting = contract.get('training_executor')
    if setting is None or setting == {'mode': 'spawn-v2', 'maximum_workers': 2}:
        return 2
    if setting != {'mode': 'spawn-v2', 'maximum_workers': 4}:
        raise ValueError('only frozen two- or four-worker training executors are supported')
    parent = contract.get('parent_campaign')
    if not isinstance(parent, dict) or Path(parent.get('path', '')).name != 'campaign.json':
        raise ValueError('four-worker training requires its frozen parent campaign')
    root = campaign.verify(parent).resolve().parent
    authority = contract.get('training_resource_authorization')
    if not isinstance(authority, dict):
        raise ValueError('four-worker training requires source-bound user resource authorization')
    document = validate_authorization(authority, root)
    if document['campaign'] != parent:
        raise ValueError('four-worker authority belongs to another frozen campaign')
    return 4


def execution_fields(root, *, training_executor=None, training_workers=None, training_resource_authorization=None, inherited=None):
    """Resolve fields for a new phase; callers seal them before launching work."""
    if training_executor not in (None, 'threads', 'spawn-v2'):
        raise ValueError('training executor must be threads or spawn-v2')
    if training_workers is not None and (type(training_workers) is not int or training_workers not in (2, 4)):
        raise ValueError('training workers must be exactly two or four')
    if training_executor is None and training_workers is None and training_resource_authorization is None:
        return {key: inherited[key] for key in ('training_executor', 'training_resource_authorization')
                if inherited is not None and key in inherited}
    mode = training_executor or ('spawn-v2' if training_workers == 4 else
                                 'spawn-v2' if inherited and inherited.get('training_executor') else 'threads')
    inherited_mode = 'spawn-v2' if inherited and inherited.get('training_executor') else 'threads'
    workers = training_workers or (expected_workers(inherited) if inherited and mode == inherited_mode else 2)
    if mode == 'threads':
        if workers != 2 or training_resource_authorization is not None:
            raise ValueError('four workers are permitted only as authorized spawned processes')
        return {}
    fields = {'training_executor': {'mode': 'spawn-v2', 'maximum_workers': workers}}
    if workers == 4:
        record = training_resource_authorization
        if record is None:
            record = inherited.get('training_resource_authorization') if inherited else None
        if record is None:
            path = authorization_path(root)
            if not path.is_file():
                raise ValueError('four-worker training requires source-bound user resource authorization')
            record = campaign.record(path)
        elif not isinstance(record, dict):
            record = campaign.record(record)
        validate_authorization(record, root)
        fields['training_resource_authorization'] = record
    elif training_resource_authorization is not None:
        raise ValueError('resource authorization cannot silently change a two-worker selection')
    return fields


def check_resume(contract, root, *, training_executor=None, training_workers=None, training_resource_authorization=None):
    expected_workers(contract)
    if training_executor is None and training_workers is None and training_resource_authorization is None:
        return
    fields = execution_fields(root, training_executor=training_executor, training_workers=training_workers,
                              training_resource_authorization=training_resource_authorization, inherited=contract)
    actual = {key: contract[key] for key in ('training_executor', 'training_resource_authorization') if key in contract}
    if fields != actual:
        raise ValueError('resume cannot change its frozen training executor or worker authorization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--authorized-at-utc')
    parser.add_argument('--user-request')
    parser.add_argument('command', choices=('authorize', 'validate'))
    args = parser.parse_args()
    if args.command == 'authorize':
        result = authorize(args.root, authorized_at_utc=args.authorized_at_utc, user_request=args.user_request)
    else:
        result = validate_authorization(campaign.record(authorization_path(args.root)), args.root)
    print(json.dumps({'maximum_seed_workers': result['training']['maximum_seed_workers'], 'new_training_started': False}))


if __name__ == '__main__':
    main()
