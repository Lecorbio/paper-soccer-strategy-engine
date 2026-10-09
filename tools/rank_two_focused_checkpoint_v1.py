"""Lossless, hash-bound float/optimizer checkpoints with bounded delta chains."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import zlib

for variable in ('MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
from tools import rank_two_focused_campaign_v1 as campaign

SCHEMA = 'papersoccer.focused-training.v1.checkpoint'
FORMAT = 'papersoccer.lossless-training-checkpoint.v1'
MAGIC = b'PSCP1\0\0\0'
MAX_BYTES = 4 * 1024**2
MAX_DEPTH = 31
DISK_RESERVE = 1024**3
LAYERS = ('w1', 'w2', 'w3')
CACHE = OrderedDict()


class DiskBudgetExceeded(RuntimeError):
    pass


def arrays(parameters, optimizer):
    if set(parameters) != set(LAYERS) or set(optimizer.first) != set(LAYERS) or set(optimizer.second) != set(LAYERS):
        raise ValueError('checkpoint layer membership differs')
    values = {name:parameters[name] for name in LAYERS}
    values.update({'first_' + name:optimizer.first[name] for name in LAYERS})
    values.update({'second_' + name:optimizer.second[name] for name in LAYERS})
    for name, value in values.items():
        if value.dtype != np.float32 or not np.all(np.isfinite(value)):
            raise ValueError('finite float32 checkpoint arrays required: ' + name)
    return values


def pack(values):
    layout, pieces, offset = [], [], 0
    for name, value in values.items():
        piece = np.asarray(value, dtype='<f4', order='C').tobytes()
        layout.append(dict(name=name, shape=list(value.shape), offset=offset, bytes=len(piece)))
        pieces.append(piece)
        offset += len(piece)
    if offset > MAX_BYTES:
        raise ValueError('checkpoint exceeds the two-family tensor bound')
    return layout, b''.join(pieces)


def validate_layout(layout, raw_bytes):
    expected = [*LAYERS, *('first_' + name for name in LAYERS), *('second_' + name for name in LAYERS)]
    if not isinstance(layout, list) or [item.get('name') for item in layout] != expected:
        raise ValueError('checkpoint array order differs')
    offset = 0
    for item in layout:
        shape = item['shape']
        if (not isinstance(shape, list) or not 1 <= len(shape) <= 2
                or any(type(value) is not int or not 0 < value <= 6301 for value in shape)
                or item['offset'] != offset or item['bytes'] != math.prod(shape) * 4):
            raise ValueError('checkpoint tensor layout differs')
        offset += item['bytes']
    if type(raw_bytes) is not int or not 0 < raw_bytes <= MAX_BYTES or offset != raw_bytes:
        raise ValueError('checkpoint raw length differs')
    for name in LAYERS:
        shapes = [item['shape'] for item in layout if item['name'] in (name, 'first_' + name, 'second_' + name)]
        if shapes.count(shapes[0]) != 3:
            raise ValueError('optimizer tensor shapes differ')


def remember(digest, layout, raw):
    CACHE[digest] = (layout, raw)
    CACHE.move_to_end(digest)
    while len(CACHE) > 2:
        CACHE.popitem(last=False)


def require_headroom(path, raw_bytes):
    parent = Path(path).resolve().parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < DISK_RESERVE + raw_bytes + 65536:
        raise DiskBudgetExceeded('less than oneGiB reserve plus a complete root checkpoint remains')


def decode(receipt, binding, chain=None):
    if receipt.get('schema') != SCHEMA or receipt.get('plan') != binding:
        raise ValueError('checkpoint/plan binding differs')
    storage = receipt.get('storage', {})
    if storage.get('format') != FORMAT or storage.get('producer') != campaign.record(__file__):
        raise ValueError('checkpoint storage producer/format differs')
    path = campaign.verify(receipt['checkpoint'])
    if campaign.read(path.with_suffix('.json')) != receipt:
        raise ValueError('checkpoint receipt differs from its retained manifest')
    digest = receipt['checkpoint']['sha256']
    chain = set() if chain is None else set(chain)
    if digest in chain or len(chain) > MAX_DEPTH:
        raise ValueError('cyclic or overlong checkpoint ancestry')
    chain.add(digest)
    validate_layout(storage['layout'], storage['raw_bytes'])
    if type(storage['depth']) is not int or not 0 <= storage['depth'] <= MAX_DEPTH:
        raise ValueError('checkpoint depth differs')
    if (storage['plan'] != binding or storage['cursor'] != receipt['cursor']
            or type(storage['optimizer_step']) is not int or storage['optimizer_step'] < 0
            or storage['optimizer_step'] != receipt['optimizer_step']):
        raise ValueError('checkpoint cursor/optimizer/plan metadata differs')
    blob = path.read_bytes()
    if not blob.startswith(MAGIC) or len(blob) < len(MAGIC) + 4:
        raise ValueError('checkpoint magic differs')
    size = struct.unpack_from('<I', blob, len(MAGIC))[0]
    if not 0 < size <= 65536 or len(blob) < len(MAGIC) + 4 + size:
        raise ValueError('checkpoint header length differs')
    inflater = zlib.decompressobj()
    header_raw = inflater.decompress(blob[len(MAGIC) + 4:len(MAGIC) + 4 + size], 65537)
    if len(header_raw) > 65536 or not inflater.eof or inflater.unused_data:
        raise ValueError('checkpoint compressed header length or termination differs')
    header = json.loads(header_raw)
    expected = {name:storage[name] for name in ('format', 'mode', 'layout', 'raw_bytes',
        'raw_sha256', 'depth', 'parent', 'optimizer_step', 'cursor', 'plan')}
    if header != expected:
        raise ValueError('checkpoint header/receipt differs')
    if storage['mode'] == 'xor':
        campaign.verify(storage['parent'])
    if digest in CACHE:
        layout, raw = CACHE[digest]
    else:
        inflater = zlib.decompressobj()
        raw = inflater.decompress(blob[len(MAGIC) + 4 + size:], storage['raw_bytes'] + 1)
        if len(raw) != storage['raw_bytes'] or not inflater.eof or inflater.unused_data:
            raise ValueError('checkpoint compressed stream length or termination differs')
        layout = storage['layout']
        if storage['mode'] == 'xor':
            if storage['parent'] is None or storage['depth'] < 1:
                raise ValueError('delta checkpoint omits its parent')
            parent = campaign.read(campaign.verify(storage['parent']))
            parent_layout, parent_raw = decode(parent, binding, chain)
            if parent_layout != layout or storage['depth'] != parent['storage']['depth'] + 1:
                raise ValueError('delta checkpoint parent layout/depth differs')
            raw = np.bitwise_xor(np.frombuffer(raw, dtype='<u4'),
                                 np.frombuffer(parent_raw, dtype='<u4')).tobytes()
        elif storage['mode'] != 'full' or storage['parent'] is not None or storage['depth'] != 0:
            raise ValueError('full checkpoint has an invalid mode or parent')
        remember(digest, layout, raw)
    if layout != storage['layout'] or hashlib.sha256(raw).hexdigest() != storage['raw_sha256']:
        raise ValueError('decoded checkpoint tensor digest differs')
    return layout, raw


def checkpoint(path, parameters, optimizer, cursor, binding, parent=None):
    path = Path(path)
    values = arrays(parameters, optimizer)
    layout, raw = pack(values)
    validate_layout(layout, len(raw))
    require_headroom(path, len(raw))
    if type(optimizer.step) not in (int, np.int64) or optimizer.step < 0:
        raise ValueError('nonnegative optimizer step required')
    common = dict(format=FORMAT, layout=layout, raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(),
                  cursor=cursor, optimizer_step=int(optimizer.step), plan=binding)
    def payload(mode, data, depth, parent_reference):
        header = dict(common, mode=mode, depth=depth, parent=parent_reference)
        encoded = zlib.compress(json.dumps(header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(), 6)
        return MAGIC + struct.pack('<I', len(encoded)) + encoded + zlib.compress(data, 6), header
    options = [payload('full', raw, 0, None)]
    if parent is not None and parent['storage']['depth'] < MAX_DEPTH:
        parent_layout, parent_raw = decode(parent, binding)
        if parent_layout != layout:
            raise ValueError('checkpoint parent tensor shapes differ')
        parent_path = Path(parent['checkpoint']['path']).with_suffix('.json')
        parent_reference = campaign.record(parent_path)
        if campaign.read(campaign.verify(parent_reference)) != parent:
            raise ValueError('checkpoint parent receipt changed')
        difference = np.bitwise_xor(np.frombuffer(raw, dtype='<u4'),
                                    np.frombuffer(parent_raw, dtype='<u4')).tobytes()
        options.append(payload('xor', difference, parent['storage']['depth'] + 1, parent_reference))
    blob, header = min(options, key=lambda item:(len(item[0]), item[1]['mode'] != 'full'))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    receipt = dict(schema=SCHEMA, checkpoint=campaign.record(path), cursor=cursor, plan=binding,
        optimizer_step=int(optimizer.step),
        parameter_sha256=hashlib.sha256(b''.join(parameters[name].tobytes() for name in LAYERS)).hexdigest(),
        storage=dict(header, producer=campaign.record(__file__), bytes=len(blob)))
    campaign.immutable(path.with_suffix('.json'), receipt)
    remember(receipt['checkpoint']['sha256'], layout, raw)
    return receipt


def restore(receipt, parameters, optimizer, binding):
    layout, raw = decode(receipt, binding)
    destinations = arrays(parameters, optimizer)
    for item in layout:
        target = destinations[item['name']]
        if list(target.shape) != item['shape']:
            raise ValueError('restored checkpoint tensor shape differs')
        value = np.frombuffer(raw, dtype='<f4', count=math.prod(item['shape']), offset=item['offset']).reshape(target.shape)
        if not np.all(np.isfinite(value)):
            raise ValueError('restored checkpoint contains nonfinite tensors')
        target[...] = value
    optimizer.step = int(receipt['optimizer_step'])
    digest = hashlib.sha256(b''.join(parameters[name].tobytes() for name in LAYERS)).hexdigest()
    if digest != receipt['parameter_sha256']:
        raise ValueError('restored parameter digest differs')
    return receipt['cursor']
