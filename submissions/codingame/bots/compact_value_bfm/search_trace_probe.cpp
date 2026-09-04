#if defined(COMPACT_VALUE_BFM_TRACE_MODULAR)
#include "engine.hpp"
#else
#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"
#endif

#include <bit>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace cv = compact_value_bfm;

namespace {

constexpr std::string_view kSchema =
    "papersoccer.compact-value-bfm-search-trace.v1";

std::string hex32(std::uint32_t value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(8) << value;
  return output.str();
}

std::string hex64(std::uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << value;
  return output.str();
}

std::string base64(const std::vector<std::uint8_t> &bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
  result.reserve(((bytes.size() + 2U) / 3U) * 4U);
  for (std::size_t offset = 0; offset < bytes.size(); offset += 3U) {
    const std::uint32_t a = bytes[offset];
    const bool have_b = offset + 1U < bytes.size();
    const bool have_c = offset + 2U < bytes.size();
    const std::uint32_t b = have_b ? bytes[offset + 1U] : 0U;
    const std::uint32_t c = have_c ? bytes[offset + 2U] : 0U;
    const std::uint32_t value = (a << 16U) | (b << 8U) | c;
    result.push_back(alphabet[(value >> 18U) & 63U]);
    result.push_back(alphabet[(value >> 12U) & 63U]);
    result.push_back(have_b ? alphabet[(value >> 6U) & 63U] : '=');
    result.push_back(have_c ? alphabet[value & 63U] : '=');
  }
  return result;
}

std::vector<std::uint8_t> pack_three_bit(
    const std::vector<std::int8_t> &weights) {
  std::vector<std::uint8_t> result((weights.size() * 3U + 7U) / 8U);
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const std::uint16_t code =
        static_cast<std::uint8_t>(weights[index]) & 7U;
    const std::size_t bit = index * 3U;
    result[bit / 8U] |= static_cast<std::uint8_t>(code << (bit % 8U));
    if (bit % 8U > 5U) {
      result[bit / 8U + 1U] |=
          static_cast<std::uint8_t>(code >> (8U - bit % 8U));
    }
  }
  return result;
}

cv::QuantizedModel zero_model() {
  return cv::QuantizedModel(cv::ModelDescriptor{
      cv::kFeatureCount, 12, 8, 1.0F, 1.0F, 1.0F, "", "", true});
}

cv::QuantizedModel patterned_model() {
  constexpr std::size_t hidden_one = 12;
  constexpr std::size_t hidden_two = 8;
  const std::size_t count = cv::kFeatureCount * hidden_one +
                            hidden_one * hidden_two + hidden_two;
  std::vector<std::int8_t> weights(count);
  std::uint64_t state = 0x243f6a8885a308d3ULL;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t mixed = state;
    mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
    mixed ^= mixed >> 31U;
    weights[index] = static_cast<std::int8_t>(mixed % 7U) - 3;
  }
  const std::vector<std::uint8_t> bytes = pack_three_bit(weights);
  const std::string encoded = base64(bytes);
  const std::string hash = cv::sha256_hex(bytes);
  return cv::QuantizedModel(cv::ModelDescriptor{
      cv::kFeatureCount, hidden_one, hidden_two,
      0.015625F, 0.03125F, 0.0625F, encoded, hash, false});
}

cv::Action parse_action(std::string_view text) {
  cv::Action action;
  for (const char character : text) {
    if (character < '0' || character > '7' ||
        action.length >= action.directions.size()) {
      throw std::runtime_error("invalid frozen transcript action");
    }
    action.directions[action.length++] =
        static_cast<std::uint8_t>(character - '0');
  }
  if (action.length == 0) {
    throw std::runtime_error("empty frozen transcript action");
  }
  return action;
}

cv::State replay(std::string_view transcript) {
  cv::State state = cv::initial_state();
  if (transcript.empty()) return state;
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t end = transcript.find('/', begin);
    const std::string_view action_text = transcript.substr(
        begin, end == std::string_view::npos ? transcript.size() - begin
                                              : end - begin);
    const cv::Action action = parse_action(action_text);
    if (state.terminal() || !cv::apply_action(state, action)) {
      throw std::runtime_error("illegal frozen transcript");
    }
    if (end == std::string_view::npos) break;
    begin = end + 1U;
  }
  if (state.terminal()) {
    throw std::runtime_error("terminal frozen search root");
  }
  return state;
}

std::string feature_hash(const cv::State &state) {
  const cv::SparseFeatures features = cv::active_features(state);
  std::vector<std::uint8_t> bytes(features.count * 2U);
  for (std::size_t index = 0; index < features.count; ++index) {
    bytes[index * 2U] = static_cast<std::uint8_t>(features.indices[index]);
    bytes[index * 2U + 1U] =
        static_cast<std::uint8_t>(features.indices[index] >> 8U);
  }
  return cv::sha256_hex(bytes);
}

struct SelectionEvidence {
  std::size_t index{};
  std::size_t tied_for_top{};
};

SelectionEvidence verify_root_selection(const cv::SearchResult &result) {
  if (result.root_actions.empty()) {
    throw std::runtime_error("search returned no root transcript");
  }
  std::size_t selected = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  std::uint32_t best_order = 0;
  for (std::size_t index = 0; index < result.root_actions.size(); ++index) {
    const cv::RootActionStat &root = result.root_actions[index];
    const double score = cv::final_score(root.value, root.visits, 1.0);
    if (index == 0 || score > best_score ||
        (score == best_score && root.order < best_order)) {
      selected = index;
      best_score = score;
      best_order = root.order;
    }
  }
  if (!(result.root_actions[selected].action == result.action)) {
    throw std::runtime_error("search action violates final-score tie-breaking");
  }
  std::size_t ties = 0;
  for (const cv::RootActionStat &root : result.root_actions) {
    ties += cv::final_score(root.value, root.visits, 1.0) == best_score;
  }
  return SelectionEvidence{selected, ties};
}

void emit_roots(const cv::SearchResult &result, const cv::State &state) {
  for (std::size_t index = 0; index < result.root_actions.size(); ++index) {
    if (index != 0) std::cout << ';';
    const cv::RootActionStat &root = result.root_actions[index];
    cv::State successor = state;
    if (!cv::apply_action(successor, root.action)) {
      throw std::runtime_error("root transcript contains an illegal turn");
    }
    std::cout << root.action.text() << ','
              << static_cast<unsigned>(root.tactical) << ','
              << hex32(std::bit_cast<std::uint32_t>(root.value)) << ','
              << hex32(std::bit_cast<std::uint32_t>(root.initial_value)) << ','
              << root.visits << ',' << root.selection_visits << ','
              << root.solved << ',' << root.order << ','
              << hex64(std::bit_cast<std::uint64_t>(
                     cv::final_score(root.value, root.visits, 1.0))) << ','
              << hex64(cv::state_hash(successor));
  }
}

void emit_stats(const cv::SearchStats &stats) {
  std::cout << stats.expansions << ',' << stats.generated_children << ','
            << stats.evaluated_children << ',' << stats.tactical_children << ','
#if !defined(COMPACT_VALUE_BFM_TRACE_LEGACY_STATS)
            << stats.cache_probes << ',' << stats.cache_hits << ','
            << stats.cache_misses << ',' << stats.widening_probes << ','
            << stats.widening_restrictions << ',' << stats.widening_eligible
            << ',' << stats.widening_deferred << ',' << stats.reuse_probes
            << ',' << stats.reuse_hits << ',' << stats.reuse_misses << ','
            << stats.reuse_rejections << ',' << stats.reused_children << ','
#endif
            << stats.generator_partial_paths << ','
            << stats.generator_proof_paths << ',' << stats.duplicate_boundaries
            << ',' << stats.fifo_extractions << ',' << stats.lifo_extractions
            << ',' << stats.tree_nodes << ',' << stats.max_depth << ','
            << stats.deadline_reached << ',' << stats.tree_cap_reached << ','
            << stats.expansion_cap_reached;
}

bool emit_case(std::size_t index, std::string_view profile,
               const cv::State &state, cv::SearchConfig config,
               const cv::QuantizedModel &model) {
  config.model = &model;
  const cv::Action emergency =
      cv::emergency_complete_action(state, config.shuffle_seed);
  if (emergency.length == 0) {
    throw std::runtime_error("frozen root has no emergency complete turn");
  }
  const cv::SearchResult result = cv::search(
      state, std::chrono::steady_clock::time_point::max(), config, &emergency);
  cv::State successor = state;
  if (result.action.length == 0 || !cv::apply_action(successor, result.action)) {
    throw std::runtime_error("search returned an illegal complete turn");
  }
  const SelectionEvidence selection = verify_root_selection(result);
  const cv::SparseFeatures features = cv::active_features(state);

  std::cout << "case"
            << "\tindex=" << index
            << "\tprofile=" << profile
            << "\tstate=" << hex64(cv::state_hash(state))
            << "\tcanonical=" << hex64(cv::canonical_state_hash(state))
            << "\tfeature=" << feature_hash(state)
            << "\tfeature_count=" << features.count
            << "\taction=" << result.action.text()
            << "\tsuccessor=" << hex64(cv::state_hash(successor))
            << "\tvalue=" << hex32(std::bit_cast<std::uint32_t>(result.value))
            << "\tsolved=" << result.solved
            << "\tselected_root=" << selection.index
            << "\ttop_ties=" << selection.tied_for_top
            << "\tlegal=1"
            << "\troot_count=" << result.root_actions.size()
            << "\tstats=";
  emit_stats(result.stats);
  std::cout << "\troots=";
  emit_roots(result, state);
  std::cout << '\n';
  return selection.tied_for_top > 1U;
}

}  // namespace

int main() {
  try {
    const cv::QuantizedModel zeros = zero_model();
    const cv::QuantizedModel patterned = patterned_model();
    cv::SearchConfig tie;
    tie.max_actions = 48;
    tie.root_partial_paths = 384;
    tie.nonroot_partial_paths = 96;
    tie.max_tree_nodes = 1'536;
    tie.max_expansions = 1;
    tie.shuffle_seed = 0x6a09e667f3bcc909ULL;
    cv::SearchConfig deep = tie;
    deep.max_expansions = 48;

    std::vector<std::string> transcripts;
    std::string line;
    while (std::getline(std::cin, line)) transcripts.push_back(line);
    if (transcripts.empty()) {
      throw std::runtime_error("empty deterministic state corpus");
    }
    std::cout << kSchema << "\tstates=" << transcripts.size()
              << "\tprofiles=2\tarchitecture=6301x12x8x1\n";
    bool saw_tie = false;
    for (std::size_t index = 0; index < transcripts.size(); ++index) {
      const cv::State state = replay(transcripts[index]);
      saw_tie = emit_case(index, "tie-root", state, tie, zeros) || saw_tie;
      emit_case(index, "patterned-deep", state, deep, patterned);
    }
    if (!saw_tie) {
      throw std::runtime_error("tie corpus did not exercise order tie-breaking");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "compact_value_bfm search trace probe failure: "
              << error.what() << '\n';
    return 1;
  }
}
