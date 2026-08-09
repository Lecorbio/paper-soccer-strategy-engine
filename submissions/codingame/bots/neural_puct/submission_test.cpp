#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "submission.cpp"
#include "visit_targets.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace np = papersoccer::neural_puct;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_close(float actual, float expected, const std::string &message) {
  require(std::abs(actual - expected) <= 1.0e-6F, message);
}

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::GameState clean_state(ps::Point point,
                          ps::Player player = ps::Player::One) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  state.ball = point;
  state.to_move = player;
  state.status = ps::Status::InProgress;
  state.path = {point};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[point] = 1;
  return state;
}

void block_except(ps::GameState &state, ps::Point point,
                  const std::vector<ps::Point> &allowed) {
  for (const ps::Point destination :
       ps::neighbors(state.config, point, state.to_move)) {
    if (std::find(allowed.begin(), allowed.end(), destination) ==
        allowed.end()) {
      state.used_segments.insert(ps::Segment{point, destination});
    }
  }
}

ps::GameState forced_rebound() {
  ps::GameState state = clean_state({4, 4});
  state.visit_count[{4, 3}] = 1;
  block_except(state, {4, 4}, {{4, 3}});
  block_except(state, {4, 3}, {{4, 4}, {4, 2}});
  return state;
}

ps::Point rotate_point(ps::Point point) {
  return {8 - point.x, 12 - point.y};
}

ps::Point reflect_point(ps::Point point) { return {8 - point.x, point.y}; }

template <typename Transform>
ps::GameState transform_state(const ps::GameState &state, Transform transform,
                              bool swap_player) {
  ps::GameState result = state;
  result.ball = transform(state.ball);
  result.to_move = swap_player ? ps::opponent(state.to_move) : state.to_move;
  result.path.clear();
  for (const ps::Point point : state.path) {
    result.path.push_back(transform(point));
  }
  result.used_segments.clear();
  for (const ps::Segment segment : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{transform(segment.a), transform(segment.b)});
  }
  result.visit_count.clear();
  for (const auto &[point, visits] : state.visit_count) {
    result.visit_count[transform(point)] = visits;
  }
  if (swap_player && result.status == ps::Status::WonByOne) {
    result.status = ps::Status::WonByTwo;
  } else if (swap_player && result.status == ps::Status::WonByTwo) {
    result.status = ps::Status::WonByOne;
  }
  return result;
}

ps::GameState asymmetric_position() {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  state = ps::apply_move(state, {{4, 5}});
  state = ps::apply_move(state, {{3, 6}});
  state = ps::apply_move(state, {{4, 6}});
  state = ps::apply_move(state, {{5, 5}});
  return state;
}

std::array<float, np::kFeatureCount> encode(const ps::GameState &state) {
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  ps::detail::SearchPosition position(topology, state);
  np::FeatureEncoder encoder(topology);
  return encoder.encode(position);
}

np::ActionFeatureMatrix encode_actions(const ps::GameState &state) {
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  ps::detail::SearchPosition position(topology, state);
  np::FeatureEncoder encoder(topology);
  return encoder.encode_actions(position);
}

void direction_and_canonical_rotation_are_exact() {
  const ps::Point center{4, 6};
  for (std::size_t direction = 0; direction < 8; ++direction) {
    const ps::Point delta = np::kDirectionDeltas[direction];
    const ps::Point destination{center.x + delta.x, center.y + delta.y};
    require(np::physical_direction(center, destination) == direction,
            "Physical direction table must round-trip.");
    require(np::decode_direction(center, static_cast<char>('0' + direction)) ==
                ps::Move{destination},
            "CodinGame direction decoding must round-trip.");
    require(np::encode_direction(center, destination) ==
                static_cast<char>('0' + direction),
            "CodinGame direction encoding must round-trip.");
    require(np::canonical_direction(center, destination, ps::Player::Two) ==
                ((direction + 4U) & 7U),
            "Player Two policy direction must rotate 180 degrees.");
  }
}

void packed_model_has_a_stable_golden_output() {
  const auto &weights = np::QuantizedModel::decoded_weights();
  require(weights.size() == np::model_data::kUnpackedWeightCount,
          "Packed int4 model must decode to the declared size.");
  require(std::all_of(weights.begin(), weights.end(),
                      [](std::int8_t value) {
                        return value >= -7 && value <= 7;
                      }),
          "Packed symmetric int4 weights must stay in [-7, 7].");
  require(!np::model_data::kModelKind.empty(),
          "Generated model must identify its artifact kind.");
  require(np::model_data::kArtifactSha256.size() == 64,
          "Generated model header must retain an artifact hash.");
  require(np::model_data::kGoldenInput ==
              (np::model_data::kActionConditionedPolicy
                   ? "stride17-mod11-action-mod13-v1"
                   : "stride17-mod11-v1"),
          "Model golden input formula must remain versioned.");

  std::array<float, np::model_data::kInputCount> features{};
  for (std::size_t index = 0; index < features.size(); index += 17) {
    features[index] = static_cast<float>((index % 11) + 1) / 11.0F;
  }
  np::ActionFeatureMatrix action_features{};
  for (std::size_t direction = 0; direction < action_features.size();
       ++direction) {
    for (std::size_t feature = 0;
         feature < action_features[direction].size(); ++feature) {
      action_features[direction][feature] =
          static_cast<float>((direction * np::kActionFeatureCount + feature) %
                                 13U +
                             1U) /
          13.0F;
    }
  }
  np::ModelOutput output;
  if constexpr (np::model_data::kActionConditionedPolicy) {
    output = np::QuantizedModel{}.evaluate(
        features, nullptr, features.size(), &action_features);
  } else {
    output = np::QuantizedModel{}.evaluate(features);
  }
  require_close(output.value, np::model_data::kGoldenValue,
                "Quantized value must match the exported golden output.");
  for (std::size_t policy = 0; policy < output.policy_logits.size(); ++policy) {
    require_close(output.policy_logits[policy],
                  np::model_data::kGoldenPolicy[policy],
                  "Quantized policy must match the exported golden output.");
    require(std::isfinite(output.policy_logits[policy]),
            "Quantized policy output must be finite.");
  }
  require(std::isfinite(output.value) && output.value >= -1.0F &&
              output.value <= 1.0F,
          "Quantized mover value must be finite and bounded.");
}

void action_features_obey_rotation_and_reflection() {
  const ps::GameState state = asymmetric_position();
  const auto normal = encode_actions(state);
  const auto rotated = encode_actions(
      transform_state(state, rotate_point, true));
  for (std::size_t direction = 0; direction < normal.size(); ++direction) {
    for (std::size_t feature = 0; feature < normal[direction].size();
         ++feature) {
      require_close(rotated[direction][feature], normal[direction][feature],
                    "Action consequences must be mover-rotation invariant.");
    }
  }

  constexpr std::array<std::size_t, 8> reflected_direction{{
      0, 7, 6, 5, 4, 3, 2, 1,
  }};
  const auto reflected = encode_actions(
      transform_state(state, reflect_point, false));
  for (std::size_t direction = 0; direction < normal.size(); ++direction) {
    const std::size_t mirrored = reflected_direction[direction];
    require_close(reflected[mirrored][0], -normal[direction][0],
                  "Reflected action x direction must change sign.");
    for (std::size_t feature = 1; feature < normal[direction].size();
         ++feature) {
      require_close(reflected[mirrored][feature], normal[direction][feature],
                    "Reflected action consequences must remap exactly.");
    }
  }
}

void initial_feature_vector_matches_the_frozen_schema() {
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const auto features = encode(state);
  require(features.size() == 1286,
          "Frozen neural PUCT feature schema must have 1,286 inputs.");
  require(std::count(features.begin() + np::kUsedEdgeOffset,
                     features.begin() + np::kDistanceOffset, 1.0F) == 0,
          "Initial state must have no used-edge flags.");
  for (std::size_t vertex = 0; vertex < 105; ++vertex) {
    const auto begin = features.begin() + np::kDistanceOffset + vertex * 8;
    require(std::count(begin, begin + 8, 1.0F) == 1,
            "Every vertex must activate exactly one distance bucket.");
  }

  const std::size_t global = np::kGlobalOffset;
  require_close(features[global + 0], 0.5F,
                "Initial canonical ball x is a schema golden.");
  require_close(features[global + 1], 0.5F,
                "Initial canonical ball y is a schema golden.");
  require_close(features[global + 2], 0.0F,
                "Initial used-edge fraction is zero.");
  require_close(features[global + 3], 1.0F / 105.0F,
                "Initial visited fraction includes only the ball.");
  require_close(features[global + 4], 1.0F,
                "Initial ball has all eight directions free.");
  require_close(features[global + 5], 4.0F / 7.0F,
                "Initial attacking-goal true-turn distance is four.");
  require_close(features[global + 6], 4.0F / 7.0F,
                "Initial own-goal true-turn distance is four.");
  require_close(features[global + 7], 1.0F / 105.0F,
                "Initial zero-cost component contains only the ball.");
  require_close(features[global + 8], 8.0F / 64.0F,
                "Initial state has eight unique safe handoffs.");
  require_close(features[global + 9], 0.0F,
                "Initial state has no dead handoff frontier.");
  for (std::size_t index = 10; index < 25; ++index) {
    require_close(features[global + index], 0.0F,
                  "Initial component/cut golden features must be zero.");
  }
}

void mover_rotation_preserves_every_feature() {
  const ps::GameState state = asymmetric_position();
  const ps::GameState rotated =
      transform_state(state, rotate_point, true);
  const auto original_features = encode(state);
  const auto rotated_features = encode(rotated);
  require(original_features == rotated_features,
          "Mover-canonical 180-degree rotation must preserve all inputs.");

  np::SearchConfig config;
  config.max_nodes = 512;
  config.max_simulations = 96;
  np::NeuralPuctSearch first(state, config);
  np::NeuralPuctSearch second(rotated, config);
  const std::vector<ps::Move> first_action = first.run();
  const std::vector<ps::Move> second_action = second.run();
  require(first_action.size() == second_action.size(),
          "Rotated deterministic search must keep action length.");
  for (std::size_t index = 0; index < first_action.size(); ++index) {
    require(rotate_point(first_action[index].to) == second_action[index].to,
            "Rotated deterministic search must rotate every action edge.");
  }
}

void horizontal_reflection_permutes_spatial_features() {
  const ps::GameState state = asymmetric_position();
  const ps::GameState reflected =
      transform_state(state, reflect_point, false);
  const auto original = encode(state);
  const auto mirrored = encode(reflected);
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);

  std::array<std::optional<ps::Segment>, 316> segments{};
  for (std::size_t source = 0; source < topology->vertex_count(); ++source) {
    for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
      const auto &adjacency = topology->adjacency(
          static_cast<ps::detail::SearchTopology::VertexIndex>(source), player);
      for (std::uint8_t index = 0; index < adjacency.count; ++index) {
        const auto &arc = adjacency.arcs[index];
        segments[arc.edge] = ps::Segment{
            topology->point(
                static_cast<ps::detail::SearchTopology::VertexIndex>(source)),
            topology->point(arc.destination)};
      }
    }
  }
  for (std::size_t edge = 0; edge < segments.size(); ++edge) {
    require(segments[edge].has_value(),
            "Reflection test must observe every edge.");
    const auto reflected_edge = topology->find_edge(
        ps::Segment{reflect_point(segments[edge]->a),
                    reflect_point(segments[edge]->b)});
    require(reflected_edge.has_value(),
            "Every edge must have a horizontal reflection.");
    require_close(original[np::kUsedEdgeOffset + edge],
                  mirrored[np::kUsedEdgeOffset + *reflected_edge],
                  "Reflection must permute used-edge inputs.");
  }
  for (std::size_t vertex = 0; vertex < 105; ++vertex) {
    const auto reflected_vertex = topology->find_vertex(reflect_point(
        topology->point(
            static_cast<ps::detail::SearchTopology::VertexIndex>(vertex))));
    require(reflected_vertex.has_value(),
            "Every vertex must have a horizontal reflection.");
    for (std::size_t bucket = 0; bucket < 8; ++bucket) {
      require_close(original[np::kDistanceOffset + vertex * 8 + bucket],
                    mirrored[np::kDistanceOffset + *reflected_vertex * 8 +
                             bucket],
                    "Reflection must permute distance buckets.");
    }
    require_close(original[np::kDegreeOffset + vertex],
                  mirrored[np::kDegreeOffset + *reflected_vertex],
                  "Reflection must permute free-degree inputs.");
  }
  require_close(original[np::kGlobalOffset + 0] +
                    mirrored[np::kGlobalOffset + 0],
                1.0F, "Reflection must flip canonical ball x.");
  for (std::size_t global = 1; global < 25; ++global) {
    require_close(original[np::kGlobalOffset + global],
                  mirrored[np::kGlobalOffset + global],
                  "Non-x globals must be reflection invariant.");
  }
}

void search_returns_complete_legal_rebound_actions() {
  const ps::GameState original = forced_rebound();
  np::SearchConfig config;
  config.max_nodes = 128;
  config.max_simulations = 32;
  np::NeuralPuctSearch search(original, config);
  const std::vector<ps::Move> action = search.run();
  require(action.size() == 2,
          "Forced two-edge rebound must be returned as one complete action.");
  ps::GameState replay = original;
  const ps::Player mover = replay.to_move;
  for (const ps::Move move : action) {
    replay = ps::apply_move(replay, move);
  }
  require(ps::is_terminal(replay) || replay.to_move != mover,
          "Returned action must stop exactly at terminal or handoff.");

  ps::GameState capped = ps::make_initial_state(codingame_rules());
  np::SearchConfig capped_config;
  capped_config.max_nodes = 1;
  capped_config.max_simulations = 8;
  np::NeuralPuctSearch capped_search(capped, capped_config);
  const std::vector<ps::Move> capped_action = capped_search.run();
  require(!capped_action.empty() && capped_search.stats().node_cap_reached,
          "Hard node cap must still return a legal NN fallback action.");

  ps::GameState expired = ps::make_initial_state(codingame_rules());
  np::SearchConfig expired_config;
  expired_config.max_nodes = 32;
  expired_config.absolute_deadline = np::SearchClock::now();
  np::NeuralPuctSearch expired_search(expired, expired_config);
  const std::vector<ps::Move> expired_action = expired_search.run();
  require(!expired_action.empty() &&
              expired_search.stats().deadline_reached &&
              expired_search.stats().simulations == 0,
          "Expired absolute deadline must skip simulations and stay legal.");
}

bool validate_visit_targets(
    ps::GameState state, const std::vector<ps::Move> &action,
    const std::vector<np::PrimitiveVisitTarget> &targets) {
  require(targets.size() == action.size(),
          "Every selected primitive edge must have one visit target.");
  bool soft = false;
  for (std::size_t edge = 0; edge < action.size(); ++edge) {
    std::array<bool, 8> legal{};
    for (const ps::Move move : ps::legal_moves(state)) {
      legal[np::canonical_direction(state.ball, move.to, state.to_move)] = true;
    }
    const std::size_t selected =
        np::canonical_direction(state.ball, action[edge].to, state.to_move);
    const np::PrimitiveVisitTarget &target = targets[edge];
    float sum = 0.0F;
    int positive = 0;
    for (std::size_t direction = 0; direction < 8; ++direction) {
      const float probability = target.probabilities[direction];
      require(std::isfinite(probability) && probability >= 0.0F,
              "Visit probabilities must be finite and nonnegative.");
      require(legal[direction] || probability == 0.0F,
              "Visit targets must assign zero mass to illegal directions.");
      require(target.probabilities[selected] + 1.0e-6F >= probability,
              "The played direction must be visit-max.");
      sum += probability;
      positive += probability > 0.0F ? 1 : 0;
    }
    require_close(sum, 1.0F, "Every visit distribution must be normalized.");
    require(target.fallback == (target.total_visits == 0),
            "Only zero-visit targets may use the one-hot fallback.");
    if (target.fallback) {
      require_close(target.probabilities[selected], 1.0F,
                    "A fallback target must be one-hot on the played edge.");
    }
    soft = soft || positive > 1;
    state = ps::apply_move(state, action[edge]);
  }
  return soft;
}

void action_visit_targets_are_soft_legal_and_behavior_neutral() {
  const ps::GameState initial = ps::make_initial_state(codingame_rules());
  np::SearchConfig plain_config;
  plain_config.max_nodes = 4'096;
  plain_config.max_simulations = 256;
  np::NeuralPuctSearch plain(initial, plain_config);
  const std::vector<ps::Move> plain_action = plain.run();

  np::SearchConfig target_config = plain_config;
  np::NeuralPuctSearch labelled(initial, target_config);
  const std::vector<ps::Move> labelled_action = labelled.run();
  const std::vector<np::PrimitiveVisitTarget> labelled_targets =
      np::NeuralPuctTrainingAccess::collect(labelled, labelled_action);
  require(labelled_action == plain_action,
          "Collecting visit targets must not alter the selected action.");
  require(labelled.stats().simulations == plain.stats().simulations &&
              labelled.stats().nodes == plain.stats().nodes &&
              labelled.stats().neural_evaluations ==
                  plain.stats().neural_evaluations,
          "Collecting targets must not alter search work.");
  require(validate_visit_targets(initial, labelled_action, labelled_targets),
          "A searched initial root should retain a genuinely soft target.");

  const ps::GameState rebound = forced_rebound();
  target_config.max_nodes = 128;
  target_config.max_simulations = 32;
  np::NeuralPuctSearch rebound_search(rebound, target_config);
  const std::vector<ps::Move> rebound_action = rebound_search.run();
  const std::vector<np::PrimitiveVisitTarget> rebound_targets =
      np::NeuralPuctTrainingAccess::collect(rebound_search, rebound_action);
  require(rebound_action.size() == 2,
          "Target test requires a two-edge complete rebound action.");
  (void)validate_visit_targets(rebound, rebound_action, rebound_targets);
}

void primitive_backup_sign_handles_rebound_and_handoff() {
  constexpr float mover_value = 0.625F;
  const float p1_leaf =
      np::mover_value_to_player_one(mover_value, ps::Player::One);
  const float p2_leaf =
      np::mover_value_to_player_one(mover_value, ps::Player::Two);
  require_close(p1_leaf, mover_value,
                "Player One mover value must already be P1-relative.");
  require_close(p2_leaf, -mover_value,
                "Player Two mover value must convert once to P1-relative.");

  require_close(np::player_one_value_for_mover(p2_leaf, ps::Player::Two),
                mover_value,
                "A same-mover Player Two rebound must not negate backup.");
  require_close(np::player_one_value_for_mover(p2_leaf, ps::Player::One),
                -mover_value,
                "A handoff to Player One must view the same leaf oppositely.");
  require_close(np::player_one_value_for_mover(p1_leaf, ps::Player::One),
                mover_value,
                "A same-mover Player One rebound must not negate backup.");
  require_close(np::player_one_value_for_mover(p1_leaf, ps::Player::Two),
                -mover_value,
                "A handoff to Player Two must view the same leaf oppositely.");
}

void exact_immediate_goal_overrides_the_placeholder() {
  ps::GameState state = clean_state({4, 1});
  block_except(state, {4, 1}, {{4, 0}});
  np::SearchConfig config;
  config.max_nodes = 16;
  config.max_simulations = 1;
  np::NeuralPuctSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  require(action.size() == 1 && action.front().to == ps::Point{4, 0},
          "Exact one-edge win must override a neutral model.");
  state = ps::apply_move(state, action.front());
  require(ps::winner(state) == ps::Player::One,
          "Immediate-goal action must have the exact rules winner.");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string, void (*)()>> tests{
      {"direction_and_canonical_rotation_are_exact",
       direction_and_canonical_rotation_are_exact},
      {"packed_model_has_a_stable_golden_output",
       packed_model_has_a_stable_golden_output},
      {"initial_feature_vector_matches_the_frozen_schema",
       initial_feature_vector_matches_the_frozen_schema},
      {"mover_rotation_preserves_every_feature",
       mover_rotation_preserves_every_feature},
      {"horizontal_reflection_permutes_spatial_features",
       horizontal_reflection_permutes_spatial_features},
      {"action_features_obey_rotation_and_reflection",
       action_features_obey_rotation_and_reflection},
      {"search_returns_complete_legal_rebound_actions",
       search_returns_complete_legal_rebound_actions},
      {"action_visit_targets_are_soft_legal_and_behavior_neutral",
       action_visit_targets_are_soft_legal_and_behavior_neutral},
      {"primitive_backup_sign_handles_rebound_and_handoff",
       primitive_backup_sign_handles_rebound_and_handoff},
      {"exact_immediate_goal_overrides_the_placeholder",
       exact_immediate_goal_overrides_the_placeholder},
  };
  for (const auto &[name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
      return 1;
    }
  }
  return 0;
}
