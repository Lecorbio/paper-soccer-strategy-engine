"""Source-bound best-first pair using the frozen incumbent four-bit evaluator."""
from __future__ import annotations

import hashlib
import importlib.util
import re
from pathlib import Path

from tools import rank_two_network32_v2 as net
from tools import rank_two_minify as minifier

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'submissions/codingame/bots/compact_value_bfm'
BOUND = {
    'engine.hpp': '6e44f83ccfa600bffe01686ebbb23e99482298622349f0952059f7604bda4bfb',
    'engine.cpp': 'c0290f6be03819f6e6b164aeaa6d567abb8f66eb47b81ba2888d1b2d051cd07a',
    'export_submission.py': '9db98098862d35efb2623acfa5848e2195f1a6e0b0cb77bd81730d5c46d9cbce',
}
SCHEMA = 'papersoccer.design-bfm.export.v1'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('bound best-first source anchor differs')
    return text.replace(old, new, 1)


def bound_sources():
    result = {}
    for name, digest in BOUND.items():
        raw = (HERE / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('historical best-first input changed: ' + name)
        result[name] = raw.decode('ascii')
    return result


PROTOCOL = r'''
namespace compact_value_bfm {
bool design_replay(int player, std::string_view history, std::string &encoded) {
  const auto turns = history.empty() ? 0U : 1U + std::count(history.begin(), history.end(), '/');
  for (const auto &row : replay_book::kReplays) {
    if (row.player_id != player || turns < row.first_turn || turns % 2U != unsigned(player)) continue;
    if (!history.empty() && (row.transcript.size() <= history.size() || !row.transcript.starts_with(history) || row.transcript[history.size()] != '/')) continue;
    if (history.empty() && row.first_turn != 0) continue;
    const auto begin = history.empty() ? 0U : history.size() + 1U;
    const auto end = row.transcript.find('/', begin);
    encoded.assign(row.transcript.substr(begin, end - begin));
    return !encoded.empty();
  }
  return false;
}
Action design_decode(std::string_view text) {
  Action action{};
  if (text.empty() || text.size() > action.directions.size()) throw std::invalid_argument("action length");
  for (const char value : text) {
    if (value < '0' || value > '7') throw std::invalid_argument("action digit");
    action.directions[action.length++] = static_cast<std::uint8_t>(value - '0');
  }
  return action;
}
}
int main() {
  using namespace compact_value_bfm;
  std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
  int player = -1; if (!(std::cin >> player)) return 0;
  if (player != 0 && player != 1) return 1;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  State state = initial_state(); bool first = true; std::string history;
  for (;;) {
    std::cin >> std::ws;
    if (std::cin.peek() == std::char_traits<char>::eof()) break;
    const auto started = std::chrono::steady_clock::now();
    auto deadline = started + std::chrono::milliseconds(first ? 990 : 180);
    if (first) {
      const auto cpu = std::clock();
      if (cpu == static_cast<std::clock_t>(-1) || cpu < 0) deadline = started;
      else deadline -= std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double, std::milli>(1000.0 * double(cpu) / CLOCKS_PER_SEC));
    }
    int length = 0; if (!(std::cin >> length)) break;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string opponent; if (!std::getline(std::cin, opponent)) break;
    try {
      if (opponent == "-") { if (player != 0 || !first) return 1; }
      else {
        if (length != int(opponent.size()) || !apply_action(state, design_decode(opponent))) return 1;
        if (!history.empty()) history += '/'; history += opponent;
      }
      if (state.terminal()) break;
      if (state.to_move != player) return 1;
      const Action fallback = emergency_complete_action(state);
      if (!fallback.length) return 1;
      Action action = fallback;
      std::string replay;
      bool corrected = false;
      if (design_replay(player, history, replay)) {
        State copy = state;
        const Action candidate = design_decode(replay);
        if (apply_action(copy, candidate)) { action = candidate; corrected = true; }
      }
      if (!corrected && std::chrono::steady_clock::now() < deadline) {
        try { action = search(state, deadline, SearchConfig{}, &fallback).action; }
        catch (const std::exception &) { action = fallback; }
      }
      if (!action.length || !apply_action(state, action)) return 1;
      const auto encoded = action.text(); std::cout << encoded << std::endl;
      if (!history.empty()) history += '/'; history += encoded; first = false;
    } catch (const std::exception &) { return 1; }
  }
  return 0;
}
'''


def export(exploration):
    if type(exploration) not in (int, float) or exploration not in (0, .95):
        raise ValueError('only the predeclared exploration pair is supported')
    sources = bound_sources()
    incumbent, codes, scales = net.baseline()
    spec = importlib.util.spec_from_file_location('design_bfm_flatten', HERE / 'export_submission.py')
    flatten = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flatten)
    markers = [incumbent.index('namespace papersoccer::turn_action_v2::' + name)
               for name in ('replay_book', 'learned_codec', 'learned_model', 'learned_eval')]
    replay = incumbent[markers[0]:markers[1]].replace('papersoccer::turn_action_v2::replay_book', 'compact_value_bfm::replay_book')
    codec = incumbent[markers[1]:markers[2]].replace('papersoccer::turn_action_v2::learned_codec', 'compact_value_bfm::codec')
    model = incumbent[markers[2]:markers[3]].replace('papersoccer::turn_action_v2::learned_model', 'compact_value_bfm::model')
    header = flatten.runtime_source_region(sources['engine.hpp'], 'papersoccer.compact-value-bfm-runtime.v1')
    cpp = flatten.runtime_source_region(sources['engine.cpp'], 'papersoccer.compact-value-bfm-runtime.v1')
    for old, new in (('kRootPartialPaths = 4\'000', "kRootPartialPaths = 50'000"),
                     ('kNonrootPartialPaths = 512', "kNonrootPartialPaths = 50'000"),
                     ("kProductionTreeNodes = 80'000", "kProductionTreeNodes = 32'768"),
                     ('kExploration = 0.95', 'kExploration = ' + str(float(exploration))),
                     ('kFinalVisitWeight = 1.0', 'kFinalVisitWeight = 0.0')):
        header = replace_once(header, old, new)
    ctor_start = cpp.index('QuantizedModel::QuantizedModel(ModelDescriptor descriptor)')
    ctor_end = cpp.index('\nPreparedEvaluation QuantizedModel::prepare', ctor_start)
    ctor = cpp[ctor_start:ctor_end]
    ctor = replace_once(ctor, 'const std::vector<std::uint8_t> bytes = decode_base64(descriptor.packed_base64);',
                        'const auto compressed = decode_base64(descriptor.packed_base64);\n  const auto bytes = codec::decode(compressed, count, 4, model::kHuffmanLengths);')
    for old, new in (('(count * 3U + 7U)', '(count * 4U + 7U)'), ('count * 3U % 8U', 'count * 4U % 8U'),
                     ('index * 3U', 'index * 4U'), ('bit % 8U > 5U', 'bit % 8U > 4U'),
                     ('& 7U', '& 15U'), ('(code & 4) != 0 ? code - 8', '(code & 8) != 0 ? code - 16'),
                     ('signed_value == -4', 'signed_value == -8')):
        ctor = replace_once(ctor, old, new)
    cpp = cpp[:ctor_start] + ctor + cpp[ctor_end:]
    start = cpp.index('    const RootTranscript *best = &roots_.front();', cpp.index('  SearchResult result() const'))
    end = cpp.index('    result.action = best->action;', start)
    cpp = cpp[:start] + '''    const RootTranscript *best = &roots_.front();
    SearchResult result;
    result.root_actions.reserve(roots_.size());
    for (const RootTranscript &root : roots_) {
      const Node &child = nodes_[root.child];
      const Node &previous = nodes_[best->child];
      const float value = -child.value;
      const bool winning = child.solved && value > 0.0F;
      const bool previous_winning = previous.solved && previous.value < 0.0F;
      if ((winning && !previous_winning) ||
          (winning == previous_winning && (value > -previous.value ||
           (value == -previous.value && (child.visits > previous.visits ||
            (child.visits == previous.visits && child.order < previous.order)))))) best = &root;
      result.root_actions.push_back(RootActionStat{root.action, root.tactical, value,
          -child.prior, child.visits, child.selection_visits, child.solved, child.order});
    }
''' + cpp[end:]
    if 'partial_paths % 10U == 0U' not in cpp or 'output_.exhaustive' not in cpp:
        raise ValueError('required hybrid traversal/unknown frontier handling absent')
    headers = {'ctime', 'iostream', 'algorithm', 'limits', 'string', 'string_view', 'stdexcept'}
    bodies = []
    for body in (codec, model, replay, header, cpp, PROTOCOL):
        kept = []
        for line in body.splitlines():
            if line.strip() == '#pragma once':
                continue
            match = re.fullmatch(r'\s*#include\s*<([^>]+)>\s*', line)
            if match:
                headers.add(match.group(1))
            elif re.fullmatch(r'\s*#include\s*"(?:engine|model)\.hpp"\s*', line):
                continue
            else:
                kept.append(line)
        bodies.append('\n'.join(kept))
    source = '\n'.join('#include <' + x + '>' for x in sorted(headers)) + '\n' + '\n'.join(bodies)
    raw = minifier.minify(source).encode('ascii')
    return raw, {'schema': SCHEMA, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                 'characters': len(raw), 'deployable_size': len(raw) < 100000,
                 'incumbent_payload_codes_sha256': hashlib.sha256(bytes(x & 255 for x in codes)).hexdigest(),
                 'fixed_scales': scales, 'architecture': [6301, 12, 8, 1], 'weight_bits': 4,
                 'clocks_ms': [990, 180], 'exploration': exploration, 'fpu': .5,
                 'max_actions': 250, 'max_partial_paths': 50000, 'max_tree_nodes': 32768,
                 'inherited_replay_book': True, 'hybrid_every_tenth_fifo': True,
                 'native_parity_verified': False, 'safety_verified': False, 'live_admitted': False,
                 'historical_inputs': BOUND.copy()}
