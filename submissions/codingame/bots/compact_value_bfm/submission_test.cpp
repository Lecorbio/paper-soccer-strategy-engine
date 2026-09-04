#define COMPACT_VALUE_BFM_NO_MAIN
#include "submission.cpp"

#include <bit>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace cv = compact_value_bfm;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void require_invalid(Function function, const char *message) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

std::string feature_sha(const cv::SparseFeatures &features) {
  std::vector<std::uint8_t> bytes(features.count * 2U);
  for (std::size_t index = 0; index < features.count; ++index) {
    bytes[index * 2U] = static_cast<std::uint8_t>(features.indices[index]);
    bytes[index * 2U + 1U] =
        static_cast<std::uint8_t>(features.indices[index] >> 8U);
  }
  return cv::sha256_hex(bytes);
}

std::string base64(const std::vector<std::uint8_t> &bytes) {
  constexpr std::string_view alphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string result;
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

std::vector<std::uint8_t> pack(const std::vector<std::int8_t> &weights) {
  std::vector<std::uint8_t> result((weights.size() * 3U + 7U) / 8U);
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const std::uint16_t code = static_cast<std::uint8_t>(weights[index]) & 7U;
    const std::size_t bit = index * 3U;
    result[bit / 8U] |= static_cast<std::uint8_t>(code << (bit % 8U));
    if (bit % 8U > 5U) {
      result[bit / 8U + 1U] |=
          static_cast<std::uint8_t>(code >> (8U - bit % 8U));
    }
  }
  return result;
}

cv::QuantizedModel patterned_model() {
  constexpr std::size_t h1 = 8;
  constexpr std::size_t h2 = 8;
  const std::size_t count = cv::kFeatureCount * h1 + h1 * h2 + h2;
  std::vector<std::int8_t> weights(count);
  for (std::size_t index = 0; index < count; ++index) {
    weights[index] = static_cast<std::int8_t>(static_cast<int>(index % 7U) - 3);
  }
  const auto bytes = pack(weights);
  const std::string encoded = base64(bytes);
  const std::string hash = cv::sha256_hex(bytes);
  return cv::QuantizedModel(cv::ModelDescriptor{
      cv::kFeatureCount, h1, h2, 0.03125F, 0.0625F, 0.125F,
      encoded, hash, false});
}

float scalar(const cv::QuantizedModel &model,
             const cv::SparseFeatures &features,
             float s1, float s2, float s3) {
  const auto weights = model.weights();
  const std::size_t h1 = model.hidden_one();
  const std::size_t h2 = model.hidden_two();
  std::array<std::int32_t, 12> sums{};
  for (std::size_t active = 0; active < features.count; ++active) {
    const std::size_t offset = features.indices[active] * h1;
    for (std::size_t hidden = 0; hidden < h1; ++hidden) {
      sums[hidden] += weights[offset + hidden];
    }
  }
  std::array<float, 12> first{};
  for (std::size_t hidden = 0; hidden < h1; ++hidden) {
    first[hidden] = cv::first_activation(static_cast<float>(sums[hidden]) * s1);
  }
  std::array<float, 16> second{};
  const std::size_t offset_two = cv::kFeatureCount * h1;
  for (std::size_t input = 0; input < h1; ++input) {
    for (std::size_t hidden = 0; hidden < h2; ++hidden) {
      volatile float scaled = first[input] * s2;
      volatile float term = scaled * weights[offset_two + input * h2 + hidden];
      second[hidden] = second[hidden] + term;
    }
  }
  for (std::size_t hidden = 0; hidden < h2; ++hidden) {
    second[hidden] = cv::second_activation(second[hidden]);
  }
  const std::size_t offset_three = offset_two + h1 * h2;
  float output = 0.0F;
  for (std::size_t hidden = 0; hidden < h2; ++hidden) {
    volatile float scaled = second[hidden] * s3;
    volatile float term = scaled * weights[offset_three + hidden];
    output = output + term;
  }
  return cv::fast_tanh(output);
}

void mark_used(cv::State &state, int edge) {
  state.used[edge >> 6] |= 1ULL << (edge & 63);
}

void keep_only(cv::State &state, int vertex,
               std::initializer_list<int> destinations) {
  const auto &topology = cv::Topology::get();
  for (int index = 0; index < topology.degree(vertex); ++index) {
    const auto arc = topology.arc(vertex, index);
    if (std::find(destinations.begin(), destinations.end(),
                  static_cast<int>(arc.destination)) == destinations.end()) {
      mark_used(state, arc.edge);
    }
  }
}

cv::State dense_rebound_state() {
  cv::State state = cv::initial_state();
  const auto &topology = cv::Topology::get();
  std::array<cv::Topology::Arc, 8> center{};
  const std::uint8_t count = cv::legal_arcs(state, center);
  for (std::uint8_t index = 0; index < count; ++index) {
    const int destination = center[index].destination;
    for (int other = 0; other < topology.degree(destination); ++other) {
      const auto arc = topology.arc(destination, other);
      if (arc.edge != center[index].edge) {
        mark_used(state, arc.edge);
        break;
      }
    }
  }
  return state;
}

void topology_and_rules_are_exact() {
  const auto &topology = cv::Topology::get();
  require(topology.vertex_at(4, 6) == 49, "canonical center index changed");
  std::array<bool, cv::kEdgeCount> edges{};
  for (int vertex = 0; vertex < cv::kVertexCount; ++vertex) {
    require(topology.rotated_vertex(topology.rotated_vertex(vertex)) == vertex,
            "vertex rotation is not involutive");
    for (int index = 0; index < topology.degree(vertex); ++index) {
      const int edge = topology.arc(vertex, index).edge;
      edges[edge] = true;
      require(topology.rotated_edge(topology.rotated_edge(edge)) == edge,
              "edge rotation is not involutive");
    }
  }
  require(std::count(edges.begin(), edges.end(), true) == cv::kEdgeCount,
          "canonical topology is not 316 edges");

  cv::State north = cv::initial_state();
  north.ball = static_cast<std::uint8_t>(topology.vertex_at(4, 1));
  require(cv::apply_edge(north, 0) && north.terminal() && north.winner == 0,
          "attacking goal rule changed");
  cv::State own = cv::initial_state();
  own.ball = static_cast<std::uint8_t>(topology.vertex_at(4, 1));
  own.to_move = 1;
  require(cv::apply_edge(own, 0) && own.terminal() && own.winner == 0,
          "own goal rule changed");

  cv::State atomic = cv::initial_state();
  cv::Action overlong;
  overlong.directions[overlong.length++] = 0;
  overlong.directions[overlong.length++] = 0;
  const cv::State snapshot = atomic;
  require(!cv::apply_action(atomic, overlong) && cv::same_boundary(atomic, snapshot),
          "rejected protocol action partially mutated compact state");
}

void features_match_the_independent_teacher() {
  cv::State initial = cv::initial_state();
  const auto features = cv::active_features(initial);
  require(features.count == 105, "initial active feature count changed");
  require(std::is_sorted(features.indices.begin(),
                         features.indices.begin() + features.count),
          "initial active features are not already canonical-sorted");
  require(feature_sha(features) ==
              "1ee1c7cf987fb1ca8d2bc707c4511b93fdf96216e6cf2a7917a18da6a2bec4d4",
          "initial features differ from independent Python teacher");
  require(cv::apply_edge(initial, 0), "north feature move failed");
  const auto after = cv::active_features(initial);
  require(after.count == 106 && feature_sha(after) ==
              "4b39d1752979d5c819cd8d4c4a022f56a0a0eb3227bd658c9fadf190dedf719f",
          "Player Two features differ from independent Python teacher");
  require(std::is_sorted(after.indices.begin(),
                         after.indices.begin() + after.count),
          "moved-state active features are not canonical-sorted");
  const cv::State rotated = cv::rotate_and_swap(initial);
  require(cv::active_features(initial) == cv::active_features(rotated),
          "mover-relative feature rotation changed");
  require(cv::canonical_state_hash(initial) == cv::canonical_state_hash(rotated),
          "mover-canonical shuffle hash changed");
}

void packed_runtime_is_strict_and_scalar_exact() {
  const cv::QuantizedModel model = patterned_model();
  cv::State state = cv::initial_state();
  const auto features = cv::active_features(state);
  const float deployed = model.evaluate(features);
  const float reference = scalar(model, features, 0.03125F, 0.0625F, 0.125F);
  require(std::bit_cast<std::uint32_t>(deployed) ==
              std::bit_cast<std::uint32_t>(reference),
          "W2/W3 scalar float32 operation order changed");
  const cv::PreparedEvaluation prepared = model.prepare(features);
  require(cv::apply_edge(state, 1), "delta fixture move failed");
  const auto next = cv::active_features(state);
  require(std::bit_cast<std::uint32_t>(model.evaluate(next)) ==
              std::bit_cast<std::uint32_t>(model.evaluate_delta(prepared, next)),
          "integer W1 full/delta inference is not bit exact");

  constexpr std::size_t count = 50'480;
  std::vector<std::int8_t> forbidden(count, 0);
  forbidden[0] = -4;
  auto bytes = pack(forbidden);
  std::string encoded = base64(bytes);
  std::string hash = cv::sha256_hex(bytes);
  require_invalid([&] {
    cv::QuantizedModel invalid(cv::ModelDescriptor{
        cv::kFeatureCount, 8, 8, 1, 1, 1, encoded, hash, false});
  }, "signed code 100 was accepted");

  std::vector<std::int8_t> capacity(75'716, 0);
  bytes = pack(capacity);
  bytes.back() |= 0xf0U;
  encoded = base64(bytes);
  hash = cv::sha256_hex(bytes);
  require_invalid([&] {
    cv::QuantizedModel invalid(cv::ModelDescriptor{
        cv::kFeatureCount, 12, 8, 1, 1, 1, encoded, hash, false});
  }, "nonzero packed padding was accepted");
  require_invalid([&] {
    cv::QuantizedModel invalid(cv::ModelDescriptor{
        cv::kFeatureCount, 8, 8, 0, 1, 1, "", "", true});
  }, "nonpositive quantization scale was accepted");
}

void generator_is_complete_tactical_and_deterministic() {
  cv::State initial = cv::initial_state();
  cv::GeneratorConfig config;
  config.max_actions = 250;
  config.max_partial_paths = 4'000;
  const auto first = cv::generate_complete_turns(initial, config);
  const auto second = cv::generate_complete_turns(initial, config);
  require(first.actions.size() == 8 && second.actions.size() == 8,
          "initial state must have eight complete turns");
  for (std::size_t index = 0; index < first.actions.size(); ++index) {
    require(first.actions[index].action == second.actions[index].action,
            "fixed-seed generator is not deterministic");
    cv::State replay = initial;
    require(cv::apply_action(replay, first.actions[index].action) &&
                cv::same_boundary(replay, first.actions[index].result),
            "generated action does not reproduce its boundary");
    for (const std::uint8_t perspective : {std::uint8_t{0}, std::uint8_t{1}}) {
      const auto child_features = cv::active_features(replay, perspective);
      require(std::is_sorted(child_features.indices.begin(),
                             child_features.indices.begin() +
                                 child_features.count),
              "generated-state active features are not canonical-sorted");
    }
  }

  cv::GeneratorConfig bounded = config;
  bounded.max_partial_paths = 20;
  const auto dense = cv::generate_complete_turns(dense_rebound_state(), bounded);
  require(dense.stats.partial_paths == 20 &&
              dense.stats.lifo_extractions == 18 &&
              dense.stats.fifo_extractions == 2,
          "complete-turn deque is not nine LIFO then one FIFO");
  require(dense.stats.proof_paths > 0 &&
              dense.stats.proof_paths <= cv::kProofPaths,
          "bounded tactical witness pass escaped its 64-path contract");
  for (std::size_t left = 0; left < dense.actions.size(); ++left) {
    for (std::size_t right = left + 1; right < dense.actions.size(); ++right) {
      require(!cv::same_boundary(dense.actions[left].result,
                                 dense.actions[right].result),
              "duplicate complete-turn boundary retained");
    }
  }

  cv::State goal = cv::initial_state();
  goal.ball = static_cast<std::uint8_t>(cv::Topology::get().vertex_at(4, 1));
  cv::GeneratorConfig one = config;
  one.max_actions = 1;
  const auto winning = cv::generate_complete_turns(goal, one);
  require(winning.actions.size() == 1 &&
              winning.actions.front().tactical == cv::TacticalClass::ImmediateGoal &&
              winning.actions.front().action.text() == "0",
          "immediate goal was not global tactical priority one");

  cv::State own = goal;
  own.to_move = 1;
  const auto own_actions = cv::generate_complete_turns(own, config);
  require(std::any_of(own_actions.actions.begin(), own_actions.actions.end(),
                      [](const auto &action) {
                        return action.tactical == cv::TacticalClass::OwnGoal &&
                               action.action.text() == "0";
                      }),
          "exact own-goal reachability witness was lost");

  cv::State danger = cv::initial_state();
  const auto &topology = cv::Topology::get();
  danger.ball = static_cast<std::uint8_t>(topology.vertex_at(4, 10));
  keep_only(danger, danger.ball, {topology.vertex_at(4, 11)});
  const auto dangerous = cv::generate_complete_turns(danger, config);
  require(dangerous.actions.size() == 1 &&
              dangerous.actions.front().tactical ==
                  cv::TacticalClass::OpponentImmediateGoal,
          "opponent immediate-goal classification changed");
}

void minimax_formulas_and_deadline_fallback_are_literal() {
  require(std::abs(cv::uct_score(0.25F, 100, 0, 0.95, 0.5) - 0.75) < 1e-12,
          "literal FPU formula changed");
  require(std::abs(cv::uct_score(0.25F, 100, 4, 0.95, 0.5) -
                   (0.25 + 0.95 * std::sqrt(std::log(100.0) / 4.0))) < 1e-12,
          "literal UCT formula changed");
  require(std::abs(cv::final_score(0.25F, 7, 1.0) -
                   (0.25 + std::log(7.0))) < 1e-12,
          "literal final visit formula changed");
  require(cv::mate_value(0, 0, 1) > cv::mate_value(0, 0, 2),
          "mate distance does not prefer the shorter win");

  cv::State state = cv::initial_state();
  const cv::Action emergency = cv::emergency_complete_action(state, 17);
  cv::SearchConfig config;
  config.max_actions = 16;
  config.root_partial_paths = 64;
  config.nonroot_partial_paths = 32;
  config.max_tree_nodes = 256;
  config.max_expansions = 4;
  config.shuffle_seed = 17;
  const cv::SearchResult expired = cv::search(
      state, std::chrono::steady_clock::time_point::min(), config, &emergency);
  require(expired.action == emergency,
          "expired root did not return the precomputed emergency action");
  require(expired.stats.deadline_reached,
          "expired search did not retain deadline telemetry");
  cv::State replay = state;
  require(cv::apply_action(replay, expired.action),
          "deadline fallback is not a legal complete turn");

  const cv::SearchResult fixed = cv::search(
      state, std::chrono::steady_clock::time_point::max(), config, &emergency);
  const cv::SearchResult repeated = cv::search(
      state, std::chrono::steady_clock::time_point::max(), config, &emergency);
  replay = state;
  require(fixed.action.length != 0 && cv::apply_action(replay, fixed.action),
          "fixed-work BFM returned an illegal complete turn");
  require(fixed.action == repeated.action && fixed.value == repeated.value &&
              fixed.root_actions.size() == repeated.root_actions.size(),
          "fixed-work descendant selection is not deterministic");
  for (std::size_t index = 0; index < fixed.root_actions.size(); ++index) {
    const auto &left = fixed.root_actions[index];
    const auto &right = repeated.root_actions[index];
    require(left.action == right.action && left.tactical == right.tactical &&
                left.value == right.value &&
                left.initial_value == right.initial_value &&
                left.visits == right.visits &&
                left.selection_visits == right.selection_visits &&
                left.solved == right.solved && left.order == right.order,
            "fixed-work root statistics are not deterministic");
  }
  require(fixed.stats.evaluated_children + fixed.stats.tactical_children ==
              fixed.stats.generated_children,
          "a generated nonsolved child escaped evaluation");
}

}  // namespace

int main() {
  try {
    topology_and_rules_are_exact();
    features_match_the_independent_teacher();
    packed_runtime_is_strict_and_scalar_exact();
    generator_is_complete_tactical_and_deterministic();
    minimax_formulas_and_deadline_fallback_are_literal();
    std::cout << "compact_value_bfm submission tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "compact_value_bfm submission test failure: "
              << error.what() << '\n';
    return 1;
  }
}
