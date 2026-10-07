#!/usr/bin/env python3
"""Versioned four-bit architecture variants on the frozen absolute-clock source."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from . import rank_two_network32_v2 as historical
except ImportError:
    import rank_two_network32_v2 as historical

codec = historical.codec
compact = historical.compact
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'papersoccer.design-network.runtime.v1'
EXPORT_SCHEMA = 'papersoccer.design-network.export.v1'
SEED = 2026093007
SOURCE_LIMIT = 100000
FEATURES = dict(historical.FEATURES)
PROFILES = {'dd8': ('distance-degree', 8, 8),
            'dd12wide': ('distance-degree', 12, 16),
            'd32': ('distance-only', 32, 32)}
HISTORICAL_SHA256 = 'de16e2fce62bd112d6fdec43cb5a611a08484289cf00c9f6136ed8f98102e1fe'
BASE_SHA256 = '78e731a81ce7cc702a200d0832a6c75e967a11b12452051724ab1387c226e4b8'
TEMPLATE_PATH = ROOT / 'results/rank_two_20260927/live-loss-v3/campaign-next-v1/calibration/180/candidate.cpp'
TEMPLATE_SHA256 = 'adda8a04d568f9cd6c4cd3e70d9f669137fec39fbb7d68a85eed4d4bf9065552'
TEMPLATE_REF = {'path': str(TEMPLATE_PATH), 'sha256': TEMPLATE_SHA256}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _source_refs():
    old_path = ROOT / 'tools/rank_two_network32_v2.py'
    if _sha(old_path.read_bytes()) != HISTORICAL_SHA256:
        raise ValueError('historical architecture producer changed')
    return {
        'producer': {'path': 'tools/rank_two_design_network_v1.py',
                     'sha256': _sha(Path(__file__).read_bytes())},
        'architecture_support': {'path': 'tools/rank_two_network32_v2.py',
                                 'sha256': HISTORICAL_SHA256},
        'incumbent': {'path': 'submissions/codingame/releases/20260927-rank3/submission.cpp',
                      'sha256': BASE_SHA256},
    }


def _shape(profile):
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError('unknown design-network profile')
    representation, h1, h2 = PROFILES[profile]
    return representation, FEATURES[representation][0], h1, h2


def _fixed_scales(scales):
    if (not isinstance(scales, (list, tuple)) or len(scales) != 3
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in scales)):
        raise ValueError('three finite positive incumbent scales required')
    try:
        normalized = [codec.f32(x) for x in scales]
    except (OverflowError, ValueError) as error:
        raise ValueError('invalid float32 scales') from error
    _, _, expected = historical.baseline()
    if normalized != expected:
        raise ValueError('fixed incumbent layer scales changed')
    return normalized


def document(profile, values, scales, *, trained=False, lineage=None):
    representation, inputs, h1, h2 = _shape(profile)
    refs = _source_refs()
    if type(trained) is not bool or (lineage is not None and not isinstance(lineage, dict)):
        raise ValueError('trained flag and lineage types changed')
    if len(values) != inputs * h1 + h1 * h2 + h2:
        raise ValueError('profile weight dimensions')
    payload = codec.pack(values, 4)
    body = {'schema': SCHEMA, 'profile': profile, 'representation': representation,
            'architecture': [inputs, h1, h2, 1], 'feature_schema': FEATURES[representation][1],
            'weight_bits': 4, 'biases': False, 'scales': _fixed_scales(scales),
            'payload_base64': base64.b64encode(payload).decode('ascii'),
            'payload_sha256': _sha(payload), 'trained': trained,
            'lineage': {} if lineage is None else lineage, 'source_refs': refs}
    # Copy nested caller-owned metadata while also rejecting non-JSON/nonfinite data.
    body = json.loads(canonical(body))
    return {**body, 'body_sha256': _sha(canonical(body))}


def validate(runtime):
    fields = {'schema', 'profile', 'representation', 'architecture', 'feature_schema',
              'weight_bits', 'biases', 'scales', 'payload_base64', 'payload_sha256',
              'trained', 'lineage', 'source_refs', 'body_sha256'}
    if not isinstance(runtime, dict) or set(runtime) != fields:
        raise ValueError('design-network runtime field roster changed')
    representation, inputs, h1, h2 = _shape(runtime['profile'])
    body = {k: v for k, v in runtime.items() if k != 'body_sha256'}
    try:
        digest = _sha(canonical(body))
    except (TypeError, ValueError) as error:
        raise ValueError('runtime is not finite canonical JSON') from error
    architecture = runtime['architecture']
    if (runtime['body_sha256'] != digest or runtime['schema'] != SCHEMA
            or architecture != [inputs, h1, h2, 1]
            or not isinstance(architecture, list) or any(type(x) is not int for x in architecture)
            or runtime['representation'] != representation
            or runtime['feature_schema'] != FEATURES[representation][1]
            or type(runtime['weight_bits']) is not int or runtime['weight_bits'] != 4
            or runtime['biases'] is not False or type(runtime['trained']) is not bool
            or not isinstance(runtime['lineage'], dict) or runtime['source_refs'] != _source_refs()):
        raise ValueError('design-network identity or dimensions differ')
    scales = _fixed_scales(runtime['scales'])
    if not isinstance(runtime['scales'], list) or scales != runtime['scales']:
        raise ValueError('runtime scales are not incumbent float32 values')
    try:
        payload = base64.b64decode(runtime['payload_base64'], validate=True)
    except (binascii.Error, TypeError, ValueError) as error:
        raise ValueError('invalid runtime payload encoding') from error
    if (_sha(payload) != runtime['payload_sha256']
            or base64.b64encode(payload).decode('ascii') != runtime['payload_base64']):
        raise ValueError('runtime payload identity differs')
    values = codec.unpack(payload, inputs * h1 + h1 * h2 + h2, 4)
    return runtime['profile'], values, scales


def initialize(profile, seed=SEED):
    _shape(profile)
    _source_refs()
    if type(seed) is not int or seed != SEED:
        raise ValueError('fixed initialization seed changed')
    if profile == 'd32':
        old = historical.initialize('d32', seed=seed)
        _, values, scales = historical.validate(old)
        lineage = {'initialization': 'historical-distance-only-embedding', 'seed': seed,
                   'historical_runtime_body_sha256': old['body_sha256'],
                   'function_preserving_initialization': False}
    else:
        _, old, scales = historical.baseline()
        if profile == 'dd8':
            values = [old[i * 12 + j] for i in range(6301) for j in range(8)]
            values += old[6301 * 12:6301 * 12 + 8 * 8] + old[-8:]
        else:
            rng = random.Random(seed)
            values = old[:6301 * 12]
            values += [old[6301 * 12 + i * 8 + j] if j < 8 else rng.choice((-1, 0, 1))
                       for i in range(12) for j in range(16)]
            values += old[-8:] + [0] * 8
        lineage = {'initialization': ('incumbent-first-eight-unit-subnetwork' if profile == 'dd8'
                                      else 'incumbent-with-seeded-extra-second-layer-units'),
                   'seed': seed, 'function_preserving_initialization': profile == 'dd12wide'}
    return document(profile, values, scales, trained=False, lineage=lineage)


def reuse_historical_d32(runtime):
    """Rebind an existing D32 payload without changing any codes, scales or training."""
    _source_refs()
    profile, values, scales = historical.validate(runtime)
    if profile != 'd32':
        raise ValueError('historical reuse requires the d32 profile')
    return document('d32', values, scales, trained=runtime['trained'], lineage={
        'reuse': 'historical-d32-exact-quantized-payload',
        'historical_runtime_body_sha256': runtime['body_sha256'],
        'historical_lineage': runtime['lineage'], 'new_training_performed': False})


def project_active(indices, representation):
    return historical.project_active(indices, representation)


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('source replacement anchor differs')
    return source.replace(old, new, 1)


def _constant(source, name, value, *, string=False):
    pattern = r'(\b' + re.escape(name) + r'\s*=(?!=)\s*)' + (r'"[^"\n]*"' if string else r'[^;\n]+') + r'(\s*;)'
    source, count = re.subn(pattern, lambda m: m[1] + value + m[2], source)
    if count != 1:
        raise ValueError('source constant anchor differs: ' + name)
    return source


def export(runtime, template_ref, clocks=(990, 180), measurement_only=False, *, minify=True):
    profile, values, scales = validate(runtime)
    if (type(measurement_only) is not bool or type(minify) is not bool
            or not isinstance(clocks, (tuple, list)) or len(clocks) != 2
            or any(type(x) is not int for x in clocks)):
        raise ValueError('explicit clock/export contract required')
    clocks = tuple(clocks)
    if clocks != (990, 180) and not (profile == 'd32' and clocks == (550, 140) and measurement_only):
        raise ValueError('550/140 is allowed only for measurement-only d32')
    if (not isinstance(template_ref, dict) or set(template_ref) != {'path', 'sha256'}
            or template_ref['sha256'] != TEMPLATE_SHA256
            or Path(template_ref['path']).resolve() != TEMPLATE_PATH.resolve()):
        raise ValueError('exact A180 template reference required')
    template = TEMPLATE_PATH.read_bytes()
    if _sha(template) != TEMPLATE_SHA256:
        raise ValueError('A180 template source changed')
    source = template.decode('ascii')
    representation, inputs, h1, h2 = _shape(profile)
    encoded, lengths, bit_count = codec.compress(values, 4)
    if codec.decompress(encoded, len(values), 4, lengths) != values:
        raise ValueError('export weight encoding failed roundtrip')
    for name, value in {'kInputs': inputs, 'kHiddenOne': h1, 'kHiddenTwo': h2,
                        'kWeightCount': len(values), 'kPackedByteCount': len(codec.pack(values, 4))}.items():
        source = _constant(source, name, str(value))
    for name, value in zip(('kScaleOne', 'kScaleTwo', 'kScaleThree'), scales):
        literal = format(value, '.9g')
        if not any(c in literal for c in '.eE'):
            literal += '.0'
        source = _constant(source, name, literal + 'F')
    for name, value in {'kFeatureSchema': FEATURES[representation][1], 'kRuntimeSchema': SCHEMA,
                        'kRuntimeBodySha256': runtime['body_sha256'],
                        'kPayloadSha256': runtime['payload_sha256'],
                        'kIdentity': 'design-network-' + profile + '-' + runtime['body_sha256'][:12]}.items():
        source = _constant(source, name, json.dumps(value), string=True)
    source, count = re.subn(r'kHuffmanLengths\{[^}]+\}',
                            'kHuffmanLengths{' + ','.join(map(str, lengths)) + '}', source)
    if count != 1:
        raise ValueError('source Huffman anchor differs')
    matches = list(historical.PAYLOAD.finditer(source))
    if len(matches) != 1:
        raise ValueError('source weight payload anchor differs')
    text = base64.b64encode(encoded).decode('ascii')
    literal = '\n'.join('"' + text[i:i + 96] + '"' for i in range(0, len(text), 96))
    match = matches[0]
    source = source[:match.start(1)] + '\n' + literal + source[match.end(1):]
    source = _replace(source,
        'static_assert(learned_model::kInputs == 6301 && learned_model::kHiddenOne == 12 &&\nlearned_model::kHiddenTwo == 8 && learned_model::kWeightBits == 4);',
        f'static_assert(learned_model::kInputs == {inputs} && learned_model::kHiddenOne == {h1} &&\nlearned_model::kHiddenTwo == {h2} && learned_model::kWeightBits == 4);')
    start = source.index('inline float evaluate(const Features &features) {')
    end = source.index('\n}   ', start)
    source = _replace(source, source[start:end], historical.EVALUATE)
    if representation == 'distance-only':
        start = source.index('int category = 56;', source.index('learned_eval::Features learned_features()'))
        end = source.index('\n}', source.index('features.indices[features.count++]', start))
        source = _replace(source, source[start:end],
            'const int category=std::min(distances_[physical],7);\nfeatures.indices[features.count++]=static_cast<std::uint16_t>(316+canonical*8+category);')
    if clocks != (990, 180):
        source = _replace(source, 'kFirstSearchTimeMs = 990;', 'kFirstSearchTimeMs = 550;')
        source = _replace(source, 'kLaterSearchTimeMs = 180;', 'kLaterSearchTimeMs = 140;')
    if minify:
        source = compact.minify(source)
    raw = source.encode('ascii')
    report = {'schema': EXPORT_SCHEMA, 'source_sha256': _sha(raw), 'characters': len(raw),
              'deployable_size': len(raw) < SOURCE_LIMIT, 'source_limit_exclusive': SOURCE_LIMIT,
              'failure': None if len(raw) < SOURCE_LIMIT else 'source-size-limit',
              'profile': profile, 'architecture': runtime['architecture'].copy(),
              'representation': representation, 'feature_schema': FEATURES[representation][1],
              'weight_bits': 4, 'biases': False, 'scales': scales, 'trained': runtime['trained'],
              'runtime_body_sha256': runtime['body_sha256'], 'payload_sha256': runtime['payload_sha256'],
              'parent': {'path': str(TEMPLATE_PATH), 'sha256': TEMPLATE_SHA256},
              'source_refs': runtime['source_refs'], 'clocks_ms': list(clocks),
              'measurement_only': measurement_only, 'live_admitted': False,
              'native_parity_verified': False, 'strength_qualified': False,
              'huffman_bits': bit_count, 'minified': minify,
              'unchanged': ['absolute-response-deadline', 'every-node-deadline-polling',
                            'startup-CPU-accounting', 'search', 'fallback', 'proofs', 'opening-book']}
    return raw, report
