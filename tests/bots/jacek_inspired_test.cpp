#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "jacek_features.hpp"
#include "jacek_neural_internal.hpp"
#include "jacek_neural_model.hpp"
#include "papersoccer/bot.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_invalid_argument(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

bool contains_move(const std::vector<ps::Move> &moves, ps::Move move) {
  return std::find(moves.begin(), moves.end(), move) != moves.end();
}

std::size_t regular_vertex(ps::Point point) {
  return static_cast<std::size_t>(point.y - 1) * 9U +
         static_cast<std::size_t>(point.x);
}

std::size_t distance_input(ps::Point point, std::size_t bucket) {
  return ps::detail::kJacekEdgeInputs +
         regular_vertex(point) * ps::detail::kJacekDistanceBuckets + bucket;
}

ps::Point rotate(ps::Point point) {
  return ps::Point{8 - point.x, 12 - point.y};
}

ps::GameState rotated_state(const ps::GameState &state) {
  ps::GameState result;
  result.config = state.config;
  result.ball = rotate(state.ball);
  result.to_move = ps::opponent(state.to_move);
  result.status = state.status;
  if (state.status == ps::Status::WonByOne) {
    result.status = ps::Status::WonByTwo;
  } else if (state.status == ps::Status::WonByTwo) {
    result.status = ps::Status::WonByOne;
  }
  for (const ps::Point point : state.path) {
    result.path.push_back(rotate(point));
  }
  for (const ps::Segment &segment : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{rotate(segment.a), rotate(segment.b)});
  }
  for (const auto &[point, visits] : state.visit_count) {
    result.visit_count.emplace(rotate(point), visits);
  }
  return result;
}

float dense_reference_logit(const ps::GameState &state) {
  const auto features = ps::detail::jacek_article_dense_features(state);
  std::array<float, 32> hidden_one =
      ps::detail::jacek_neural_model::kB1;
  for (std::size_t input = 0; input < features.size(); ++input) {
    if (features[input] == 0) {
      continue;
    }
    for (std::size_t hidden = 0; hidden < hidden_one.size(); ++hidden) {
      hidden_one[hidden] +=
          ps::detail::jacek_neural_model::kW1[input * 32U + hidden];
    }
  }
  for (float &value : hidden_one) {
    value = std::max(value, 0.0F);
  }
  std::array<float, 32> hidden_two =
      ps::detail::jacek_neural_model::kB2;
  for (std::size_t first = 0; first < hidden_one.size(); ++first) {
    for (std::size_t second = 0; second < hidden_two.size(); ++second) {
      hidden_two[second] += hidden_one[first] *
          ps::detail::jacek_neural_model::kW2[first * 32U + second];
    }
  }
  for (float &value : hidden_two) {
    value = std::max(value, 0.0F);
  }
  float result = ps::detail::jacek_neural_model::kB3;
  for (std::size_t hidden = 0; hidden < hidden_two.size(); ++hidden) {
    result += hidden_two[hidden] *
              ps::detail::jacek_neural_model::kW3[hidden];
  }
  return result;
}

void initial_features_match_the_article_contract() {
  const auto features =
      ps::detail::jacek_article_dense_features(ps::make_initial_state());
  require(features.size() == 1156,
          "Jacek features should contain exactly 1,156 inputs.");
  require(std::all_of(
              features.begin(),
              features.begin() + ps::detail::kJacekEdgeInputs,
              [](std::uint8_t value) { return value == 0; }),
          "The initial position should not activate a drawn-edge input.");

  std::array<std::size_t, 8> histogram{};
  std::size_t active = 0;
  for (std::size_t vertex = 0;
       vertex < ps::detail::kJacekVertices; ++vertex) {
    std::size_t vertex_active = 0;
    for (std::size_t bucket = 0; bucket < 8; ++bucket) {
      const std::uint8_t value =
          features[316U + vertex * 8U + bucket];
      active += value;
      vertex_active += value;
      histogram[bucket] += value;
    }
    require(vertex_active == 1,
            "Each vertex should activate exactly one distance bucket.");
  }
  require(active == 105 &&
              histogram == std::array<std::size_t, 8>{
                  1, 8, 16, 46, 28, 6, 0, 0},
          "Initial true-turn distances should match the article fixture.");
}

void true_turn_distance_uses_rebounds_and_boundaries() {
  ps::GameState rebound = ps::make_initial_state();
  rebound.ball = ps::Point{3, 6};
  rebound.path = {rebound.ball};
  rebound.used_segments.clear();
  rebound.visit_count.clear();
  rebound.visit_count[rebound.ball] = 1;
  rebound.visit_count[ps::Point{4, 5}] = 1;
  const auto rebound_features =
      ps::detail::jacek_article_dense_features(rebound);
  require(rebound_features[distance_input(ps::Point{4, 5}, 0)] == 1 &&
              rebound_features[distance_input(ps::Point{2, 5}, 1)] == 1,
          "Visited destinations should cost zero and fresh nodes one turn.");

  ps::GameState boundary = ps::make_initial_state();
  boundary.ball = ps::Point{1, 6};
  boundary.path = {boundary.ball};
  boundary.used_segments.clear();
  boundary.visit_count.clear();
  boundary.visit_count[boundary.ball] = 1;
  const auto boundary_features =
      ps::detail::jacek_article_dense_features(boundary);
  require(boundary_features[distance_input(ps::Point{0, 6}, 0)] == 1 &&
              boundary_features[distance_input(ps::Point{1, 5}, 1)] == 1,
          "An unvisited boundary should be reachable in the current turn.");
}

void player_two_rotation_is_bit_and_logit_exact() {
  ps::GameState state = ps::make_initial_state();
  state.ball = ps::Point{3, 5};
  state.to_move = ps::Player::One;
  state.path = {ps::Point{4, 6}, ps::Point{3, 5}};
  state.used_segments = {
      ps::Segment{ps::Point{4, 6}, ps::Point{3, 5}},
      ps::Segment{ps::Point{3, 5}, ps::Point{2, 5}},
      ps::Segment{ps::Point{2, 5}, ps::Point{2, 4}},
  };
  state.visit_count.clear();
  state.visit_count[ps::Point{4, 6}] = 1;
  state.visit_count[ps::Point{3, 5}] = 2;
  state.visit_count[ps::Point{2, 5}] = 1;
  state.visit_count[ps::Point{2, 4}] = 1;
  const ps::GameState twin = rotated_state(state);

  require(ps::detail::jacek_article_dense_features(state) ==
              ps::detail::jacek_article_dense_features(twin),
          "Rotating the board and mover should preserve every canonical input.");
  require(ps::detail::jacek_article_mover_logit(state) ==
              ps::detail::jacek_article_mover_logit(twin),
          "Canonical twins should accumulate the network in identical order.");
}

void sparse_inference_matches_a_dense_reference() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{ps::Point{3, 5}});
  state = ps::apply_move(state, ps::Move{ps::Point{2, 5}});
  const float sparse = ps::detail::jacek_article_mover_logit(state);
  const float dense = dense_reference_logit(state);
  require(std::abs(sparse - dense) <= 1e-6F,
          "Sparse binary inference should match the dense 1,156-input network.");
}

void non_empty_position_matches_the_reviewed_feature_and_model_fixture() {
  ps::GameState state = ps::make_initial_state();
  for (const ps::Point destination : {
           ps::Point{3, 5},
           ps::Point{2, 5},
           ps::Point{3, 6},
           ps::Point{4, 6},
       }) {
    state = ps::apply_move(state, ps::Move{destination});
  }
  require(state.ball == ps::Point{4, 6} &&
              state.to_move == ps::Player::Two &&
              state.used_segments.size() == 4,
          "The golden transcript should end in a Player Two rebound.");

  const ps::detail::JacekSparseFeatures features =
      ps::detail::jacek_article_sparse_features(state);
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  for (std::size_t index = 0; index < features.count; ++index) {
    const std::uint16_t active = features.indices[index];
    hash ^= active & 0xffU;
    hash *= 0x100000001b3ULL;
    hash ^= active >> 8U;
    hash *= 0x100000001b3ULL;
  }
  require(features.count == 109 && hash == 0xf90097cecb3bbbb8ULL,
          "The reviewed fixture should preserve canonical edge and distance ordering.");

  const float logit = ps::detail::jacek_article_mover_logit(state);
  require(std::abs(logit - 0.0194436312F) <= 2e-6F,
          "Non-empty C++ inference should match independent float32 JSON inference.");
}

void checkpoint_matches_python_reference() {
  const float logit =
      ps::detail::jacek_article_mover_logit(ps::make_initial_state());
  require(std::abs(logit - 0.0383620039F) <= 2e-6F,
          "C++ inference should match the model JSON's float32 reference logit.");
  require(ps::JacekInspiredBot::model_sha256() ==
              "57412763f650350a1036e438a7a18656c3da675a2f27c7308001acfb12407084",
          "The embedded model identity should match the reviewed checkpoint.");
}

void bot_is_trained_legal_and_deterministic() {
  require(ps::JacekInspiredBot::model_sha256().size() == 64 &&
              ps::JacekInspiredBot::model_sha256() != "untrained",
          "The demo bot should embed a trained, identified checkpoint.");
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 2;
  config.max_nodes = 5'000;
  config.transposition_table_entries = 2'048;
  ps::JacekInspiredBot first(config);
  ps::JacekInspiredBot second(config);
  const ps::GameState state = ps::make_initial_state();
  const ps::Move first_move = first.choose_move(state);
  const ps::Move second_move = second.choose_move(state);
  require(first.name() == "JacekInspiredBot" &&
              first_move == second_move &&
              contains_move(ps::legal_moves(state), first_move),
          "The trained Jacek-inspired bot should be deterministic and legal.");
  require(first.last_search_stats().nodes > 0 &&
              first.last_search_stats().leaf_evaluations > 0,
          "The Jacek-inspired bot should expose its shared alpha-beta work.");
}

void bot_takes_immediate_goals_for_both_players() {
  ps::AlphaBetaConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 1'000;
  ps::JacekInspiredBot bot(config);

  ps::GameState one = ps::make_initial_state();
  one.ball = ps::Point{4, 1};
  one.to_move = ps::Player::One;
  one.path = {one.ball};
  one.used_segments.clear();
  one.visit_count = {{one.ball, 1}};
  require(bot.choose_move(one) == ps::Move{ps::Point{4, 0}},
          "Player One should take an immediate north goal.");

  ps::GameState two = ps::make_initial_state();
  two.ball = ps::Point{4, 11};
  two.to_move = ps::Player::Two;
  two.path = {two.ball};
  two.used_segments.clear();
  two.visit_count = {{two.ball, 1}};
  require(bot.choose_move(two) == ps::Move{ps::Point{4, 12}},
          "Player Two should take an immediate south goal.");
}

void bot_rejects_incompatible_rules_and_factory_preserves_limits() {
  ps::BotConfig config;
  config.kind = ps::BotKind::JacekInspired;
  config.alpha_beta_depth = 2;
  config.alpha_beta_max_nodes = 4'000;
  config.alpha_beta_transposition_table_entries = 1'024;
  config.alpha_beta_max_search_plies = 9;
  config.alpha_beta_max_time_ms = 17;
  std::unique_ptr<ps::Bot> base = ps::make_bot(config);
  const auto *bot = dynamic_cast<const ps::JacekInspiredBot *>(base.get());
  require(bot != nullptr && base->name() == "JacekInspiredBot" &&
              bot->config().max_turn_depth == 2 &&
              bot->config().max_nodes == 4'000 &&
              bot->config().transposition_table_entries == 1'024 &&
              bot->config().max_search_plies == 9 &&
              bot->config().max_time_ms == 17 &&
              ps::bot_kind_name(ps::BotKind::JacekInspired) ==
                  "JacekInspiredBot",
          "The public factory should construct the configured demo bot.");

  ps::RulesConfig wrong_size;
  wrong_size.width = 6;
  require_invalid_argument(
      [&] {
        ps::JacekInspiredBot invalid;
        (void)invalid.choose_move(ps::make_initial_state(wrong_size));
      },
      "The fixed article network should reject a different board topology.");

  ps::RulesConfig wrong_rules;
  wrong_rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  wrong_rules.blocked_rule = ps::BlockedRule::MoverLoses;
  require_invalid_argument(
      [&] {
        ps::JacekInspiredBot invalid;
        (void)invalid.choose_move(ps::make_initial_state(wrong_rules));
      },
      "The model should reject game rules outside its training contract.");
}

}  // namespace

int run_jacek_inspired_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"initial_features_match_the_article_contract",
       initial_features_match_the_article_contract},
      {"true_turn_distance_uses_rebounds_and_boundaries",
       true_turn_distance_uses_rebounds_and_boundaries},
      {"player_two_rotation_is_bit_and_logit_exact",
       player_two_rotation_is_bit_and_logit_exact},
      {"sparse_inference_matches_a_dense_reference",
       sparse_inference_matches_a_dense_reference},
      {"non_empty_position_matches_the_reviewed_feature_and_model_fixture",
       non_empty_position_matches_the_reviewed_feature_and_model_fixture},
      {"checkpoint_matches_python_reference",
       checkpoint_matches_python_reference},
      {"bot_is_trained_legal_and_deterministic",
       bot_is_trained_legal_and_deterministic},
      {"bot_takes_immediate_goals_for_both_players",
       bot_takes_immediate_goals_for_both_players},
      {"bot_rejects_incompatible_rules_and_factory_preserves_limits",
       bot_rejects_incompatible_rules_and_factory_preserves_limits},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
    } catch (...) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": unknown error\n";
    }
  }
  std::cout << '\n' << (tests.size() - static_cast<std::size_t>(failures))
            << '/' << tests.size() << " Jacek-inspired tests passed.\n";
  return failures == 0 ? 0 : 1;
}
