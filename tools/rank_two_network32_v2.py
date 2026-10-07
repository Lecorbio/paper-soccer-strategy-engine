#!/usr/bin/env python3
"""Matched three-arm evaluator contracts; initialization is not training."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys
import random

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from . import compact_representation as codec, rank_two_minify as compact
except ImportError:
    import compact_representation as codec, rank_two_minify as compact

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'submissions/codingame/releases/20260927-rank3/submission.cpp'
BASE_SHA = '78e731a81ce7cc702a200d0832a6c75e967a11b12452051724ab1387c226e4b8'
SCHEMA = 'papersoccer.network32.runtime.v2'
FEATURES = {
    'distance-degree': (6301, 'papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:mover-relative-rotate180:true-turn-distance+free-degree'),
    'distance-only': (1156, 'papersoccer.network32.distance-only.v1:edge316+vertex105x8:mover-relative-rotate180:true-turn-distance'),
}
PROFILES = {'dd12': ('distance-degree', 12, 8), 'd12': ('distance-only', 12, 8),
            'd32': ('distance-only', 32, 32)}
PAYLOAD = re.compile(r'kPackedWeights\s*=\s*((?:\s*"[^"\n]*"\s*)+);')


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('ascii')


def baseline():
    raw = BASE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_SHA:
        raise ValueError('frozen baseline source changed')
    source = raw.decode('ascii')
    def number(name):
        return int(re.search(r'\b' + name + r'\s*=\s*(\d+)\s*;', source)[1])
    if [number(k) for k in ('kInputs', 'kHiddenOne', 'kHiddenTwo', 'kWeightBits')] != [6301, 12, 8, 4]:
        raise ValueError('baseline architecture differs')
    match = PAYLOAD.search(source)
    packed = base64.b64decode(''.join(re.findall(r'"([^"\n]*)"', match[1])), validate=True)
    lengths = list(map(int, re.search(r'kHuffmanLengths\{([^}]+)\}', source)[1].split(',')))
    values = codec.decompress(packed, number('kWeightCount'), 4, lengths)
    scales = [codec.f32(float(re.search(r'\b' + k + r'\s*=\s*([^;]+);', source)[1].strip().rstrip('F')))
              for k in ('kScaleOne', 'kScaleTwo', 'kScaleThree')]
    return source, values, scales


def project_active(indices, representation):
    """Preserve edge/vertex identities; distance projection drops degree only."""
    if representation not in FEATURES:
        raise ValueError('unknown representation')
    if list(indices) != sorted(set(indices)) or any(type(x) is not int or not 0 <= x < 6301 for x in indices):
        raise ValueError('invalid canonical active inputs')
    vertices = [(x - 316) // 57 for x in indices if x >= 316]
    if vertices != list(range(105)):
        raise ValueError('exactly one category per canonical vertex required')
    if representation == 'distance-degree':
        return list(indices)
    return [x if x < 316 else 316 + ((x - 316) // 57) * 8 + ((x - 316) % 57) // 8 for x in indices]


def projected_key(indices, representation):
    try:
        from . import jacek_replay_features as features
    except ImportError:
        import jacek_replay_features as features
    normal = project_active(indices, representation)
    mirrored = project_active(features.reflect_active(indices), representation)
    return hashlib.sha256(canonical(min(normal, mirrored))).hexdigest()


def projection_collisions(rows, representation):
    """Keep every root sharing a reflected projected feature key across splits."""
    seen = {}
    for row in rows:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('protected/test rows are not accepted')
        key = projected_key(row['active'], representation)
        splits = seen.setdefault(key, {'train': set(), 'validation': set()})
        splits[row['split']].add(row['root_id'])
    return [{'feature_key': key, 'train_roots': sorted(splits['train']),
             'validation_roots': sorted(splits['validation'])}
            for key, splits in sorted(seen.items()) if all(splits.values())]


def document(profile, values, scales, *, trained=False, lineage=None):
    representation, h1, h2 = PROFILES[profile]
    inputs, feature_schema = FEATURES[representation]
    if len(values) != inputs * h1 + h1 * h2 + h2:
        raise ValueError('profile weight dimensions')
    scales = [codec.f32(x) for x in scales]
    if len(scales) != 3 or any(not 0 < x < float('inf') for x in scales):
        raise ValueError('finite positive layer scales required')
    packed = codec.pack(values, 4)
    body = {'schema': SCHEMA, 'profile': profile, 'representation': representation, 'architecture': [inputs, h1, h2, 1],
            'feature_schema': feature_schema, 'weight_bits': 4, 'biases': False,
            'scales': scales, 'payload_base64': base64.b64encode(packed).decode('ascii'),
            'trained': bool(trained), 'lineage': lineage or {}, 'base_source_sha256': BASE_SHA}
    return {**body, 'body_sha256': hashlib.sha256(canonical(body)).hexdigest()}


def validate(runtime):
    body = {k: v for k, v in runtime.items() if k != 'body_sha256'}
    profile = body.get('profile')
    if profile not in PROFILES or body.get('schema') != SCHEMA:
        raise ValueError('network32 contract')
    representation, h1, h2 = PROFILES[profile]
    inputs, schema = FEATURES[representation]
    if (body.get('architecture') != [inputs, h1, h2, 1] or body.get('representation') != representation or body.get('feature_schema') != schema
            or body.get('weight_bits') != 4 or body.get('biases') is not False
            or body.get('base_source_sha256') != BASE_SHA
            or hashlib.sha256(canonical(body)).hexdigest() != runtime.get('body_sha256')):
        raise ValueError('network32 identity or dimensions differ')
    scales = body['scales']
    if len(scales) != 3 or any(not 0 < x < float('inf') for x in scales):
        raise ValueError('invalid scales')
    values = codec.unpack(base64.b64decode(body['payload_base64'], validate=True), inputs * h1 + h1 * h2 + h2, 4)
    return profile, values, scales


def initialize(profile, seed=2026093007):
    """Embed the deployed model; seeded extra units initially have zero output paths."""
    representation, h1, h2 = PROFILES[profile]
    _, old, scales = baseline()
    first = [old[i * 12:i * 12 + 12] for i in range(6301)]
    if representation == 'distance-only':
        projected = first[:316]
        for vertex in range(105):
            for distance in range(8):
                rows = [first[316 + vertex * 57 + c] for c in
                        ([56] if distance == 7 else range(distance * 8, distance * 8 + 8))]
                projected.append([int(round(sum(row[j] for row in rows) / len(rows))) for j in range(12)])
        first = projected
    elif representation != 'distance-degree':
        raise ValueError('unknown representation')
    # Existing paths preserve the incumbent output. Distinct extra units avoid
    # permanently symmetric training gradients; their output weights start at zero.
    rng = random.Random(seed)
    values = [row[j] if j < 12 else rng.choice((-1, 0, 1)) for row in first for j in range(h1)]
    offset = 6301 * 12
    values += [old[offset + i * 8 + j] if i < 12 and j < 8 else
               (0 if j < 8 else rng.choice((-1, 0, 1))) for i in range(h1) for j in range(h2)]
    values += old[-8:] + [0] * (h2 - 8)
    return document(profile, values, scales, lineage={
        'initialization': 'embedded-current-model-and-seeded-extra-hidden-units', 'seed': seed,
        'function_preserving': representation == 'distance-degree',
        'distance_only_projection': 'mean degree rows per distance; integer ties-to-even',
        'training_performed': False})


EVALUATE = '''inline float evaluate(const Features &features) {
const auto &model = weights();
constexpr auto H1=learned_model::kHiddenOne,H2=learned_model::kHiddenTwo;
std::array<std::int32_t,H1> first{};
for(std::size_t i=0;i<features.count;++i){
const auto offset=features.indices[i]*H1;
for(std::size_t j=0;j<H1;++j)first[j]+=static_cast<int>(model[offset+j]);
}
std::array<float,H1> activated{};
for(std::size_t i=0;i<H1;++i){const float x=static_cast<float>(first[i])*learned_model::kScaleOne;
activated[i]=x<0.0F?0.01F*x:x*x;}
std::array<float,H2> second{};
constexpr auto offset_two=learned_model::kInputs*H1;
for(std::size_t i=0;i<H1;++i)for(std::size_t j=0;j<H2;++j){
volatile float scaled=activated[i]*learned_model::kScaleTwo;
volatile float term=scaled*static_cast<float>(model[offset_two+i*H2+j]);
second[j]=second[j]+term;}
for(auto &x:second)x=x<0.0F?0.01F*x:x;
constexpr auto offset_three=offset_two+H1*H2;
float output=0.0F;
for(std::size_t j=0;j<H2;++j){volatile float scaled=second[j]*learned_model::kScaleThree;
volatile float term=scaled*static_cast<float>(model[offset_three+j]);output=output+term;}
return fast_tanh(output);
}'''


def export(runtime, *, minify=False):
    profile, values, scales = validate(runtime)
    representation, h1, h2 = PROFILES[profile]
    source, _, _ = baseline()
    inputs = FEATURES[representation][0]
    encoded, lengths, bit_count = codec.compress(values, 4)
    if codec.decompress(encoded, len(values), 4, lengths) != values:
        raise ValueError('weight encoding failed roundtrip')
    for key, value in {'kInputs': inputs, 'kHiddenOne': h1, 'kHiddenTwo': h2,
                       'kWeightCount': len(values), 'kPackedByteCount': len(codec.pack(values, 4))}.items():
        source, count = re.subn(r'(\b' + key + r'\s*=\s*)\d+(\s*;)', lambda m: m[1] + str(value) + m[2], source)
        if count != 1: raise ValueError('source constant anchor')
    for key, value in zip(('kScaleOne', 'kScaleTwo', 'kScaleThree'), scales):
        literal = format(value, '.9g')
        if not any(c in literal for c in '.eE'): literal += '.0'
        source, count = re.subn(r'(\b' + key + r'\s*=\s*)[^;]+;', lambda m: m[1] + literal + 'F;', source)
        if count != 1: raise ValueError('source scale anchor')
    for key, value in {'kFeatureSchema': FEATURES[representation][1], 'kRuntimeSchema': SCHEMA,
                       'kRuntimeBodySha256': runtime['body_sha256'],
                       'kPayloadSha256': hashlib.sha256(codec.pack(values, 4)).hexdigest(),
                       'kIdentity': 'network32-research-' + runtime['body_sha256'][:12]}.items():
        source, count = re.subn(r'(\b' + key + r'\s*=\s*)"[^"\n]*";', lambda m: m[1] + json.dumps(value) + ';', source)
        if count != 1: raise ValueError('source metadata anchor')
    source = re.sub(r'kHuffmanLengths\{[^}]+\}', 'kHuffmanLengths{' + ','.join(map(str, lengths)) + '}', source)
    match = PAYLOAD.search(source)
    text = base64.b64encode(encoded).decode('ascii')
    literal = '\n'.join('"' + text[i:i + 96] + '"' for i in range(0, len(text), 96))
    source = source[:match.start(1)] + '\n' + literal + source[match.end(1):]
    source = codec.incumbent.replace_once(source,
        'static_assert(learned_model::kInputs == 6301 && learned_model::kHiddenOne == 12 &&\nlearned_model::kHiddenTwo == 8 && learned_model::kWeightBits == 4);',
        f'static_assert(learned_model::kInputs == {inputs} && learned_model::kHiddenOne == {h1} &&\nlearned_model::kHiddenTwo == {h2} && learned_model::kWeightBits == 4);')
    start = source.index('inline float evaluate(const Features &features) {')
    end = source.index('\n}   ', start)
    source = source[:start] + EVALUATE + source[end:]
    if representation == 'distance-only':
        start = source.index('int category = 56;', source.index('learned_eval::Features learned_features()'))
        end = source.index('\n}', source.index('features.indices[features.count++]', start))
        source = source[:start] + 'const int category=std::min(distances_[physical],7);\nfeatures.indices[features.count++]=static_cast<std::uint16_t>(316+canonical*8+category);' + source[end:]
    if minify: source = compact.minify(source)
    raw = source.encode('ascii')
    report = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'characters': len(raw),
              'deployable_size': len(raw) < 100000, 'model': runtime['body_sha256'],
              'profile': profile, 'representation': representation, 'architecture': runtime['architecture'],
              'trained': runtime['trained'], 'initialization_is_strength_evidence': False,
              'live_admitted': False, 'huffman_bits': bit_count, 'minified': minify,
              'base_source_sha256': BASE_SHA,
              'unchanged': ['search', '550/140ms', 'replay corrections', 'fallback', 'proof separation']}
    return raw, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=PROFILES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minify', action='store_true')
    args = parser.parse_args()
    runtime = initialize(args.profile)
    source, report = export(runtime, minify=args.minify)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('runtime.json', canonical(runtime)), ('submission.cpp', source), ('report.json', canonical(report))]:
        path = args.output / name
        path.write_bytes(data); path.chmod(0o444)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__': main()
