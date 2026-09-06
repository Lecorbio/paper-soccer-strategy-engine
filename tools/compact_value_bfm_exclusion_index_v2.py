#!/usr/bin/env python3
"""Explicit future-only packed exclusion sets with source-bound mmap loading.

Build one source document at a time, sort bounded runs of full 256-bit keys,
then merge them without retaining the entire union as Python strings/sets.
Workers hash the original sources without parsing their fingerprint arrays.
No caller or historical phase opts in automatically; the new plan must bind the
returned index record and this loader's source. Callers own the heavy-stage lease.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import heapq
import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compact_value_bfm_campaign_v2 as campaign

SCHEMA = campaign.ID + '.packed-base-exclusions.v2'
SOURCE_VERSION = 'full-sha256-role-unions-v1'
HEX = re.compile(r'[0-9a-f]{64}')
RECORD_BYTES = 32
CHUNK_RECORDS = 65536
MERGE_FAN_IN = 16


def _canonical(path):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('packed exclusion path must be canonical and not redirected')
    return path


def _bound(record, expected):
    expected = _canonical(expected)
    if Path(record.get('path', '')).absolute() != expected:
        raise ValueError('packed exclusion artifact left its explicit canonical path')
    return campaign.verify(record)


def _sources():
    from tools import compact_value_bfm_stream_v2 as stream
    return {'producer': campaign.record(Path(__file__)), 'legacy_reader': campaign.record(Path(campaign.__file__)),
            'membership_loader': campaign.record(Path(stream.__file__))}


def _key(role, domain):
    if not isinstance(role, str) or not role or not isinstance(domain, str) or not domain:
        raise ValueError('exclusion role and domain must be nonempty strings')
    return role, domain


def _source_runs(binding, directory):
    """Only this source's JSON object and one bounded raw-byte buffer coexist."""
    source = campaign.read(campaign.verify(binding))
    key = _key(source.get('role'), source.get('domain'))
    fingerprints = source.get('fingerprints')
    if not isinstance(fingerprints, list):
        raise ValueError('source exclusions must contain a fingerprint list')
    paths = []; buffer = []
    for value in fingerprints:
        if not isinstance(value, str) or HEX.fullmatch(value) is None:
            raise ValueError('source exclusion is not a full canonical lowercase SHA256')
        buffer.append(bytes.fromhex(value))
        if len(buffer) == CHUNK_RECORDS:
            path = directory / f'run-{len(paths):06d}.bin'
            _write_run(path, buffer); paths.append(path); buffer.clear()
    if buffer:
        path = directory / f'run-{len(paths):06d}.bin'
        _write_run(path, buffer); paths.append(path)
    return key, len(fingerprints), paths


def _write_run(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        for value in sorted(set(values)):
            # Never convert fixed-width NumPy scalars to bytes: trailing NULs
            # are genuine hash bytes, not string padding to trim.
            if len(value) != RECORD_BYTES:
                raise ValueError('packed exclusion key lost its full 256 bits')
            output.write(value)


def _records(file):
    while True:
        value = file.read(RECORD_BYTES)
        if not value:
            return
        if len(value) != RECORD_BYTES:
            raise ValueError('partial key in packed exclusion run')
        yield value


def _merge(paths, output):
    if len(paths) > MERGE_FAN_IN:
        raise ValueError('packed exclusion merge exceeded its bounded fan-in')
    count = 0; previous = None
    with contextlib.ExitStack() as opened:
        streams = [_records(opened.enter_context(path.open('rb'))) for path in paths]
        target = opened.enter_context(output.open('xb'))
        for value in heapq.merge(*streams):
            if value != previous:
                target.write(value); count += 1; previous = value
        target.flush(); os.fsync(target.fileno())
    return count


def _union(paths, directory):
    directory.mkdir(parents=True, exist_ok=True)
    generation = 0
    while len(paths) > MERGE_FAN_IN:
        merged = []
        for offset in range(0, len(paths), MERGE_FAN_IN):
            path = directory / f'merge-{generation:04d}-{offset:06d}.bin'
            _merge(paths[offset:offset + MERGE_FAN_IN], path); merged.append(path)
        for path in paths:
            path.unlink()
        paths = merged; generation += 1
    final = directory / 'union.bin'
    count = _merge(paths, final)
    return final, count


def _publish(source, destination):
    """Publish a durable file once without copying its whole payload into RAM."""
    destination = _canonical(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        if destination.is_symlink() or (destination.stat().st_size, campaign.sha(destination)) != (
                source.stat().st_size, campaign.sha(source)):
            raise ValueError('immutable packed exclusion array differs')
    return campaign.record(destination)


def _array_path(index_path, ordinal):
    return index_path.parent / (index_path.name + '.arrays') / f'{ordinal:04d}.bin'


def build_index(contract_path, index_path):
    """Build an explicit new index; existing index/source bytes are never replaced."""
    contract_path, index_path = _canonical(contract_path), _canonical(index_path)
    contract_record = campaign.record(contract_path)
    if index_path.exists():
        record = campaign.record(index_path)
        validate_index(record, contract_record=contract_record)
        return record
    contract = campaign.read(contract_path)
    exclusions = contract['exclusions']
    if not isinstance(exclusions, list):
        raise ValueError('contract exclusions must be an ordered list')
    sources = _sources(); groups = OrderedDict(); source_rows = []
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.packed-exclusions-', dir=index_path.parent) as temporary:
        temporary = Path(temporary)
        for ordinal, binding in enumerate(exclusions):
            key, count, runs = _source_runs(binding, temporary / f'source-{ordinal:06d}')
            group = groups.setdefault(key, {'runs': [], 'source_ordinals': [], 'input_fingerprints': 0})
            group['runs'].extend(runs); group['source_ordinals'].append(ordinal)
            group['input_fingerprints'] += count
            source_rows.append({'ordinal': ordinal, 'source': binding, 'role': key[0], 'domain': key[1],
                                'input_fingerprints': count})
        entries = []
        for ordinal, ((role, domain), group) in enumerate(groups.items()):
            path, count = _union(group['runs'], temporary / f'union-{ordinal:06d}')
            entries.append({'ordinal': ordinal, 'role': role, 'domain': domain,
                'source_ordinals': group['source_ordinals'], 'input_fingerprints': group['input_fingerprints'],
                'unique_fingerprints': count, 'array': _publish(path, _array_path(index_path, ordinal))})
        # A changed source during construction cannot produce a completed index.
        campaign.verify(contract_record)
        for binding in exclusions:
            campaign.verify(binding)
        campaign.seal(index_path, {'schema': SCHEMA, 'source_version': SOURCE_VERSION,
            'contract': contract_record, 'exclusion_sources': exclusions, 'source_rows': source_rows,
            'sources': sources, 'entries': entries, 'record_bytes': RECORD_BYTES,
            'ordering': 'first-seen-role-domain;sorted-unique-full-32-byte-keys',
            'build_memory': 'one-source-json-document-plus-bounded-sort-and-merge-buffers',
            'contains_labels': False, 'contains_metrics': False, 'contains_transcripts': False,
            'historical_exclusions_modified': False})
    record = campaign.record(index_path)
    validate_index(record, contract_record=contract_record)
    return record



def _validate_array(record, expected_path, expected_count):
    path = _canonical(expected_path)
    if Path(record.get('path', '')).absolute() != path:
        raise ValueError('packed exclusion array left its explicit canonical path')
    digest = hashlib.sha256(); count = 0; previous = None; size = 0
    with path.open('rb') as source:
        while block := source.read(CHUNK_RECORDS * RECORD_BYTES):
            digest.update(block); size += len(block)
            if len(block) % RECORD_BYTES:
                raise ValueError('packed exclusion array contains a partial SHA256')
            for offset in range(0, len(block), RECORD_BYTES):
                value = block[offset:offset + RECORD_BYTES]
                if previous is not None and value <= previous:
                    raise ValueError('packed exclusion array must be strictly sorted and unique')
                previous = value; count += 1
    if (record != {'path': str(path), 'bytes': size, 'sha256': digest.hexdigest()}
            or count != expected_count):
        raise ValueError('packed exclusion array hash/count/size binding changed')
    return path

def validate_index(index_record, *, contract_record):
    """Verify the frozen contract, source versions and every source/array hash.

    Source files are hashed in streaming chunks, never loaded as JSON in workers.
    Their role/count projections come from the source-bound builder descriptor.
    """
    path = _canonical(index_record['path']); _bound(index_record, path)
    document = campaign.read(path)
    contract_path = _canonical(contract_record['path']); _bound(contract_record, contract_path)
    contract = campaign.read(contract_path)
    if (document.get('schema') != SCHEMA or document.get('source_version') != SOURCE_VERSION
            or document.get('contract') != contract_record or document.get('sources') != _sources()
            or document.get('record_bytes') != RECORD_BYTES
            or document.get('ordering') != 'first-seen-role-domain;sorted-unique-full-32-byte-keys'
            or document.get('build_memory') != 'one-source-json-document-plus-bounded-sort-and-merge-buffers'
            or any(document.get(key) is not False for key in ('contains_labels', 'contains_metrics',
                'contains_transcripts', 'historical_exclusions_modified'))
            or document.get('exclusion_sources') != contract['exclusions']):
        raise ValueError('packed exclusions changed their source version, contract or semantics')
    rows = document.get('source_rows')
    if not isinstance(rows, list) or len(rows) != len(contract['exclusions']):
        raise ValueError('packed exclusion source roster changed')
    groups = OrderedDict()
    for ordinal, (row, binding) in enumerate(zip(rows, contract['exclusions'], strict=True)):
        if (set(row) != {'ordinal', 'source', 'role', 'domain', 'input_fingerprints'}
                or type(row['ordinal']) is not int or row['ordinal'] != ordinal or row['source'] != binding
                or type(row['input_fingerprints']) is not int or row['input_fingerprints'] < 0):
            raise ValueError('packed exclusion source projection changed')
        campaign.verify(binding)
        key = _key(row['role'], row['domain'])
        group = groups.setdefault(key, {'source_ordinals': [], 'input_fingerprints': 0})
        group['source_ordinals'].append(ordinal); group['input_fingerprints'] += row['input_fingerprints']
    entries = document.get('entries')
    if not isinstance(entries, list) or len(entries) != len(groups):
        raise ValueError('packed exclusion role/domain roster changed')
    for ordinal, (entry, (key, group)) in enumerate(zip(entries, groups.items(), strict=True)):
        if (set(entry) != {'ordinal', 'role', 'domain', 'source_ordinals', 'input_fingerprints',
                'unique_fingerprints', 'array'} or type(entry['ordinal']) is not int or entry['ordinal'] != ordinal
                or (entry['role'], entry['domain']) != key
                or entry['source_ordinals'] != group['source_ordinals']
                or any(type(value) is not int for value in entry['source_ordinals'])
                or entry['input_fingerprints'] != group['input_fingerprints']
                or type(entry['unique_fingerprints']) is not int
                or not 0 <= entry['unique_fingerprints'] <= group['input_fingerprints']
                or (entry['unique_fingerprints'] == 0) != (group['input_fingerprints'] == 0)):
            raise ValueError('packed exclusion union/order/count changed')
        _validate_array(entry['array'], _array_path(path, ordinal), entry['unique_fingerprints'])
    return document


class PackedFingerprintIndex:
    """Exact lowercase-hex set membership using the existing mmap lookup."""
    def __init__(self, path):
        from tools.compact_value_bfm_stream_v2 import FingerprintIndex
        self._lookup = FingerprintIndex(path)
        self._lookup.values.flags.writeable = False

    def __contains__(self, fingerprint):
        if not isinstance(fingerprint, str) or HEX.fullmatch(fingerprint) is None:
            return False
        return fingerprint in self._lookup

    def __len__(self):
        return len(self._lookup.values)


def load_index(index_record, *, contract_record):
    document = validate_index(index_record, contract_record=contract_record)
    return {(entry['role'], entry['domain']): PackedFingerprintIndex(entry['array']['path'])
            for entry in document['entries']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'validate'))
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--index', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        with campaign.lease(args.contract.resolve().parent):
            record = build_index(args.contract, args.index)
    else:
        record = campaign.record(args.index)
        validate_index(record, contract_record=campaign.record(args.contract))
    print(campaign.raw({'index': record, 'source_version': SOURCE_VERSION,
                       'historical_exclusions_modified': False}).decode(), end='')


if __name__ == '__main__':
    main()
