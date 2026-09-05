#!/usr/bin/env python3
"""Deterministic diagnostic-only timing instrumentation of frozen compact sources.

The derivative is never a deployment source or a speed-gate binary. Each entry
and exit samples steady_clock and assigns elapsed wall time to the innermost
active category. Nested scopes, including recursion within one category, are
exclusive: their durations are not counted twice. Timer calls, bookkeeping,
altered compiler inlining and any OS preemption remain in these measurements;
no overhead estimate is subtracted. Pair the derivative with the original fixed
probe to verify its search trace and measure the observed total-time overhead.

Only the known minified generated source signatures below are supported. An
expected complete source SHA256 and unique code anchors are mandatory. All
original bytes remain in order; removing the recorded additions recovers the
original exactly. Model payload, source file and deployable source size are not
modified. The manifest records function body hashes for independent auditing.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'papersoccer.compact-timing-instrumentation.v2'
PROBE_SCHEMA = 'papersoccer.compact-engine-category-probe.v2'
VERSION = 'exclusive-function-entry-v1'
CATEGORIES = (
    'feature_construction', 'first_layer', 'dense_evaluation',
    'complete_turn_generation', 'tree_traversal', 'state_application',
    'residual_search',
)
# Whole signatures end at the opening brace; declarations never match.
ANCHORS = (
    ('feature_construction', 'SparseFeatures active_features(const State&state,std::uint8_t perspective){'),
    ('first_layer', 'PreparedEvaluation QuantizedModel::prepare(const SparseFeatures&features)const noexcept{'),
    ('first_layer', 'float QuantizedModel::evaluate_delta(const PreparedEvaluation&base,const SparseFeatures&features)const noexcept{'),
    ('dense_evaluation', 'float QuantizedModel::finish(std::array<std::int32_t,12>first)const noexcept{'),
    ('complete_turn_generation', 'Action emergency_complete_action(const State&source,std::uint64_t seed){'),
    ('complete_turn_generation', 'GenerationResult generate_complete_turns(const State&state,const GeneratorConfig&config){'),
    ('tree_traversal', 'bool xD(int current,std::vector<int> &path){'),
    ('tree_traversal', 'std::optional<std::vector<int>>xE(){'),
    ('tree_traversal', 'void refresh(int index){'),
    ('tree_traversal', 'SearchResult result()const{'),
    ('state_application', 'bool apply_edge(State&state,std::uint8_t direction)noexcept{'),
    ('state_application', 'bool apply_action(State&state,const Action&action,bool require_complete)noexcept{'),
)
DESCRIPTIONS = {
    'feature_construction': 'Perspective-aware active_features, including distance search and optional feature sorting.',
    'first_layer': 'prepare and evaluate_delta: sparse accumulation, feature differences and copies; excludes nested finish.',
    'dense_evaluation': 'finish: first activation, dense second/output layers, second activation and output tanh.',
    'complete_turn_generation': 'Emergency action and complete-turn generator, including tactics, paths and ordering; excludes nested state application.',
    'tree_traversal': 'Descendant path selection (xE/xD), refresh/backup and final root selection; recursive scopes counted exclusively.',
    'state_application': 'apply_action and apply_edge, including validation and state updates; nested edges counted exclusively.',
    'residual_search': 'All remaining interval time: initialization/model loading when first used, tree allocation/expansion bookkeeping, evaluation dispatch, destruction and selected action encoding.',
}
INTERVAL = 'emergency_complete_action+search+selected-action-encoding;excludes-root-replay-and-post-search-invariant-checks'

RUNTIME = r'''// Diagnostic-only derivative; original source remains unchanged.
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
namespace cvbfm_timing_v2 {
inline constexpr char source_sha256[] = "__SOURCE_SHA256__";
inline constexpr char instrumentation_version[] = "exclusive-function-entry-v1";
enum class Category : std::size_t { feature_construction, first_layer,
  dense_evaluation, complete_turn_generation, tree_traversal, state_application,
  residual_search, count };
inline constexpr std::size_t count = static_cast<std::size_t>(Category::count);
inline constexpr std::array<const char*, count> names{{"feature_construction",
  "first_layer", "dense_evaluation", "complete_turn_generation", "tree_traversal",
  "state_application", "residual_search"}};
using Clock = std::chrono::steady_clock;
struct Totals { std::array<std::uint64_t, count> ns{}, calls{};
  std::uint64_t total_search_ns{}, category_sum_ns{}; };
class Scope;
inline thread_local Totals totals;
inline thread_local Scope* current = nullptr;
inline thread_local Clock::time_point started, last;
inline thread_local bool enabled = false;
class Scope {
 public:
  Category category;
  Scope* parent;
  bool active;
  static void charge(Clock::time_point now) noexcept {
    const auto which = current == nullptr ? Category::residual_search : current->category;
    const auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now-last).count();
    if (duration < 0) std::abort();
    totals.ns[static_cast<std::size_t>(which)] += static_cast<std::uint64_t>(duration);
    last = now;
  }
  explicit Scope(Category value) noexcept : category(value), parent(current), active(enabled) {
    if (!active) return;
    charge(Clock::now());
    current = this;
    ++totals.calls[static_cast<std::size_t>(category)];
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
  ~Scope() noexcept {
    if (!active) return;
    if (current != this) std::abort();
    charge(Clock::now());
    current = parent;
  }
};
inline void begin() noexcept {
  if (enabled || current != nullptr) std::abort();
  totals = {};
  totals.calls[static_cast<std::size_t>(Category::residual_search)] = 1;
  started = last = Clock::now();
  enabled = true;
}
inline Totals end() noexcept {
  if (!enabled || current != nullptr) std::abort();
  const auto now = Clock::now();
  Scope::charge(now);
  enabled = false;
  totals.total_search_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now-started).count());
  for (auto value : totals.ns) totals.category_sum_ns += value;
  if (totals.total_search_ns != totals.category_sum_ns) std::abort();
  return totals;
}
}
'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_mask(text: str) -> str:
    """Mask ordinary C++ comments/literals while retaining byte positions."""
    result = list(text)
    index = 0
    while index < len(text):
        start = index
        if text.startswith('//', index):
            end = text.find('\n', index)
            index = len(text) if end < 0 else end
        elif text.startswith('/*', index):
            end = text.find('*/', index + 2)
            if end < 0:
                raise ValueError('unterminated C++ comment')
            index = end + 2
        elif text.startswith('R"', index):
            raise ValueError('raw C++ strings are not supported by the frozen source transformer')
        elif text[index] in ('"', "'"):
            quote = text[index]
            # The generated source uses C++ digit separators (4'000 etc.).
            if (quote == "'" and index and index + 1 < len(text)
                    and text[index - 1].isdigit() and text[index + 1].isdigit()):
                index += 1
                continue
            index += 1
            while index < len(text):
                if text[index] == '\\':
                    index += 2
                elif text[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError('unterminated C++ literal')
        else:
            index += 1
            continue
        result[start:index] = ' ' * (index - start)
    return ''.join(result)


def instrument_source(source: bytes, expected_source_sha256: str) -> tuple[bytes, dict]:
    if (not isinstance(source, bytes) or not re.fullmatch('[0-9a-f]{64}', expected_source_sha256)
            or sha(source) != expected_source_sha256):
        raise ValueError('instrumentation requires the expected exact source SHA256')
    text = source.decode('ascii')
    if 'cvbfm_timing_v2' in text:
        raise ValueError('source already contains diagnostic timing instrumentation')
    for token in ('kInputs=6301;', 'kHiddenOne=12;', 'kHiddenTwo=8;', 'kOutputs=1;'):
        if text.count(token) != 1:
            raise ValueError('source does not have the fixed scalar architecture')
    payloads = re.findall(r'kPayloadSha256="([0-9a-f]{64})";', text)
    weights = re.findall(r'kPackedWeights=((?:"[A-Za-z0-9+/=]*")+);', text)
    if len(payloads) != 1 or len(weights) != 1:
        raise ValueError('source must contain one exact packed model')
    if sha(base64.b64decode(weights[0].replace('"', ''), validate=True)) != payloads[0]:
        raise ValueError('source packed payload identity mismatch')
    mask = code_mask(text)
    insertions = []
    for ordinal, (category, anchor) in enumerate(ANCHORS):
        if text.count(anchor) != 1 or mask.count(anchor) != 1:
            raise ValueError('timing function anchor must occur exactly once: ' + anchor)
        start = mask.index(anchor)
        position = start + len(anchor)
        depth, end = 1, position
        while depth and end < len(mask):
            depth += (mask[end] == '{') - (mask[end] == '}')
            end += 1
        if depth:
            raise ValueError('unbalanced timing function body')
        addition = ('::cvbfm_timing_v2::Scope cvbfm_timing_scope_' + str(ordinal)
                    + '(::cvbfm_timing_v2::Category::' + category + ');')
        insertions.append({'category': category, 'anchor': anchor, 'count': 1,
                           'offset': position, 'addition': addition,
                           'original_function_sha256': sha(source[start:end])})
    insertions.sort(key=lambda item: item['offset'])
    derivative = text
    for item in reversed(insertions):
        derivative = derivative[:item['offset']] + item['addition'] + derivative[item['offset']:]
    runtime = RUNTIME.replace('__SOURCE_SHA256__', expected_source_sha256)
    derivative = (runtime + derivative).encode('ascii')
    # Prove the transformer touched no original source/model byte.
    recovered = derivative[len(runtime):].decode('ascii')
    for item in insertions:
        if recovered.count(item['addition']) != 1:
            raise ValueError('ambiguous timing insertion')
        recovered = recovered.replace(item['addition'], '', 1)
    if recovered.encode('ascii') != source:
        raise ValueError('instrumentation changed original source bytes')
    return derivative, {
        'schema': SCHEMA, 'instrumentation_version': VERSION,
        'source_sha256': expected_source_sha256, 'source_bytes': len(source),
        'instrumented_source_sha256': sha(derivative), 'instrumented_source_bytes': len(derivative),
        'payload_sha256': payloads[0], 'runtime_sha256': sha(runtime.encode('ascii')),
        'runtime_bytes': len(runtime), 'anchors': insertions,
        'categories': [{'name': name, 'description': DESCRIPTIONS[name]} for name in CATEGORIES],
        'interval': INTERVAL, 'exclusive': True, 'clock': 'std::chrono::steady_clock',
        'original_source_recoverable': True, 'original_model_bytes_unchanged': True,
        'attribution_only': True, 'eligible_for_speed_gate': False,
        'overhead': 'Timer calls, bookkeeping, compiler effects and preemption included; no correction. Compare original fixed probe total time; never use derivative for throughput or p95 gates.',
    }


def validate_probe_parity(baseline: dict, attributed: dict, manifest: dict) -> dict:
    """Reject changed behavior, identity, unaccounted time or malformed timers."""
    if (manifest.get('schema') != SCHEMA or manifest.get('instrumentation_version') != VERSION
            or manifest.get('categories') != [{'name': name, 'description': DESCRIPTIONS[name]} for name in CATEGORIES]
            or manifest.get('attribution_only') is not True or manifest.get('eligible_for_speed_gate') is not False
            or baseline.get('schema') != 'papersoccer.compact-engine-version-probe.v2'
            or attributed.get('schema') != PROBE_SCHEMA
            or attributed.get('source_sha256') != manifest.get('source_sha256')
            or attributed.get('instrumentation_version') != VERSION
            or attributed.get('attribution_only') is not True or attributed.get('eligible_for_speed_gate') is not False):
        raise ValueError('timing source, schema or attribution-only binding differs')
    invariants = ('all_actions_legal', 'all_root_actions_legal',
                  'actual_model_full_delta_bit_exact', 'all_root_actions_full_delta_bit_exact')
    for raw in (baseline, attributed):
        if (raw.get('mode') != 'fixed' or raw.get('payload_sha256') != manifest['payload_sha256']
                or any(raw.get(key) is not True for key in invariants)
                or not isinstance(raw.get('rows'), list) or not raw['rows']):
            raise ValueError('fixed-work payload, roots or invariant checks differ')
    if len(baseline['rows']) != len(attributed['rows']):
        raise ValueError('fixed-work root counts differ')
    seen = set()
    counters = ('nodes', 'expansions', 'generated_successors', 'evaluated_successors')
    for original, row in zip(baseline['rows'], attributed['rows']):
        for key in ('id', 'action', 'fixed_trace', *counters):
            if key not in original or row.get(key) != original[key]:
                raise ValueError('instrumentation changed fixed-work trace: ' + key)
        if (not isinstance(row['id'], str) or not row['id'] or row['id'] in seen
                or not isinstance(row['action'], str) or not row['action']
                or set(row['action']) - set('01234567')
                or not isinstance(row['fixed_trace'], str) or not row['fixed_trace']):
            raise ValueError('invalid fixed-work row identity/action/trace')
        seen.add(row['id'])
        for raw in (original, row):
            if (any(type(raw[key]) is not int or raw[key] < 0 for key in counters)
                    or isinstance(raw.get('milliseconds'), bool)
                    or not isinstance(raw.get('milliseconds'), (float, int))
                    or not math.isfinite(raw['milliseconds']) or raw['milliseconds'] <= 0):
                raise ValueError('invalid native fixed-work counters or duration')
        ns, calls = row.get('category_exclusive_ns'), row.get('category_calls')
        if (not isinstance(ns, dict) or not isinstance(calls, dict)
                or set(ns) != set(CATEGORIES) or set(calls) != set(CATEGORIES)
                or any(type(value) is not int or value < 0 for value in (*ns.values(), *calls.values()))
                or type(row.get('total_search_ns')) is not int or row['total_search_ns'] <= 0
                or type(row.get('category_sum_ns')) is not int
                or row['category_sum_ns'] != sum(ns.values())
                or row['category_sum_ns'] != row['total_search_ns']
                or row.get('reconciled') is not True or calls['residual_search'] != 1
                or any(ns[name] > 0 and calls[name] == 0 for name in CATEGORIES)
                or not math.isclose(row['milliseconds'], row['total_search_ns'] / 1e6,
                                    rel_tol=1e-12, abs_tol=1e-9)):
            raise ValueError('exclusive timing categories do not reconcile')
    return {'fixed_trace_bit_exact': True, 'all_category_totals_reconciled': True,
            'rows': len(seen), 'attribution_only': True, 'eligible_for_speed_gate': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    if len({path.resolve() for path in (args.source, args.output, args.manifest)}) != 3:
        raise ValueError('source, derivative and manifest must be separate paths')
    derivative, manifest = instrument_source(args.source.read_bytes(), args.expected_source_sha256)
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode('ascii')
    for path, data in ((args.output, derivative), (args.manifest, encoded)):
        if path.exists() and path.read_bytes() != data:
            raise ValueError('refusing to overwrite different diagnostic evidence: ' + str(path))
    for path, data in ((args.output, derivative), (args.manifest, encoded)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(data)


if __name__ == '__main__':
    main()
