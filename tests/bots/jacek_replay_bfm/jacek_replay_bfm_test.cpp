#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"
#include "papersoccer/bot.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_invalid(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

ps::RulesConfig codingame_rules() {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  return rules;
}

void write_u32(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint32_t value) {
  for (std::size_t index = 0; index < 4U; ++index) {
    bytes[offset + index] =
        static_cast<std::uint8_t>(value >> (index * 8U));
  }
}

void write_u64(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint64_t value) {
  write_u32(bytes, offset, static_cast<std::uint32_t>(value));
  write_u32(bytes, offset + 4U, static_cast<std::uint32_t>(value >> 32U));
}

std::uint8_t hex_digit(char value) {
  if (value >= '0' && value <= '9') {
    return static_cast<std::uint8_t>(value - '0');
  }
  return static_cast<std::uint8_t>(value - 'a' + 10);
}

void write_hash(std::vector<std::uint8_t> &bytes, std::size_t offset,
                std::string_view hash) {
  require(hash.size() == 64U, "Test checkpoint hash must be SHA-256.");
  for (std::size_t index = 0; index < 32U; ++index) {
    bytes[offset + index] = static_cast<std::uint8_t>(
        (hex_digit(hash[index * 2U]) << 4U) |
        hex_digit(hash[index * 2U + 1U]));
  }
}

std::vector<std::uint8_t> make_checkpoint(
    const std::vector<std::pair<std::size_t, float>> &weights = {}) {
  const std::size_t payload_bytes =
      ps::detail::kReplayBfmWeightCount * sizeof(float);
  std::vector<std::uint8_t> bytes(
      ps::detail::kReplayBfmRuntimeHeaderBytes + payload_bytes, 0U);
  const std::array<std::uint8_t, 8> magic{
      'J', 'R', 'B', 'F', 'M', 0, 0, 1};
  std::copy(magic.begin(), magic.end(), bytes.begin());
  write_u32(bytes, 8U, ps::detail::kReplayBfmRuntimeHeaderBytes);
  write_u32(bytes, 12U, 1U);
  write_u32(bytes, 16U, ps::detail::kReplayBfmInputCount);
  write_u32(bytes, 20U, ps::detail::kReplayBfmHiddenOne);
  write_u32(bytes, 24U, ps::detail::kReplayBfmHiddenTwo);
  write_u32(bytes, 28U, 1U);
  write_u32(bytes, 32U, 1U);
  write_u32(bytes, 36U, 2U);
  write_u32(bytes, 40U, 3U);
  write_u32(bytes, 44U, std::bit_cast<std::uint32_t>(0.01F));
  write_u64(bytes, 48U, ps::detail::kReplayBfmWeightCount);
  write_hash(bytes, 56U, ps::detail::kReplayBfmFeatureSchemaSha256);

  for (const auto &[index, value] : weights) {
    require(index < ps::detail::kReplayBfmWeightCount,
            "Test weight index must be in range.");
    write_u32(bytes,
              ps::detail::kReplayBfmRuntimeHeaderBytes + index * 4U,
              std::bit_cast<std::uint32_t>(value));
  }
  const std::span<const std::uint8_t> payload(
      bytes.data() + ps::detail::kReplayBfmRuntimeHeaderBytes,
      payload_bytes);
  write_hash(bytes, 88U, ps::detail::replay_bfm_sha256_hex(payload));
  return bytes;
}

class TemporaryCheckpoint {
 public:
  explicit TemporaryCheckpoint(const std::vector<std::uint8_t> &bytes) {
    static std::uint64_t sequence = 0;
    const auto stamp = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path_ = std::filesystem::temp_directory_path() /
            ("papersoccer-jacek-replay-bfm-" + std::to_string(stamp) + "-" +
             std::to_string(sequence++) + ".runtime");
    std::ofstream output(path_, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    require(static_cast<bool>(output), "Could not write test checkpoint.");
  }

  ~TemporaryCheckpoint() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  std::string string() const { return path_.string(); }

 private:
  std::filesystem::path path_;
};

ps::Point rotate(ps::Point point) {
  return ps::Point{8 - point.x, 12 - point.y};
}

ps::GameState rotated_state(const ps::GameState &state) {
  ps::GameState result;
  result.config = state.config;
  result.ball = rotate(state.ball);
  result.to_move = ps::opponent(state.to_move);
  result.status = state.status;
  if (result.status == ps::Status::WonByOne) {
    result.status = ps::Status::WonByTwo;
  } else if (result.status == ps::Status::WonByTwo) {
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

ps::GameState replay_complete_turn_prefix(std::string_view prefix) {
  static constexpr std::array<ps::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
  }};
  ps::GameState state = ps::make_initial_state(codingame_rules());
  std::size_t begin = 0U;
  for (;;) {
    const std::size_t end = prefix.find('/', begin);
    const std::string_view action = prefix.substr(
        begin, end == std::string_view::npos ? prefix.size() - begin
                                             : end - begin);
    require(!action.empty(), "Regression prefix contains an empty turn.");
    const ps::Player mover = state.to_move;
    for (const char direction : action) {
      require(direction >= '0' && direction <= '7',
              "Regression prefix contains an invalid direction.");
      const ps::Point delta =
          deltas[static_cast<std::size_t>(direction - '0')];
      state = ps::apply_move(
          state, ps::Move{{state.ball.x + delta.x, state.ball.y + delta.y}});
    }
    require(ps::is_terminal(state) || state.to_move != mover,
            "Regression prefix contains an incomplete turn.");
    if (end == std::string_view::npos) {
      return state;
    }
    begin = end + 1U;
  }
}

bool same_capped_search_stats(const ps::JacekReplayBfmSearchStats &left,
                              const ps::JacekReplayBfmSearchStats &right) {
  return left.expansions == right.expansions &&
         left.generated_actions == right.generated_actions &&
         left.retained_actions == right.retained_actions &&
         left.neural_evaluations == right.neural_evaluations &&
         left.visits == right.visits &&
         left.completed_actions == right.completed_actions &&
         left.duplicate_boundaries == right.duplicate_boundaries &&
         left.partial_paths == right.partial_paths &&
         left.fifo_extractions == right.fifo_extractions &&
         left.lifo_extractions == right.lifo_extractions &&
         left.tactical_proofs == right.tactical_proofs &&
         left.tactical_solutions == right.tactical_solutions &&
         left.truncations == right.truncations &&
         left.generation_action_cap_stops ==
             right.generation_action_cap_stops &&
         left.generation_partial_cap_stops ==
             right.generation_partial_cap_stops &&
         left.generation_deadline_stops ==
             right.generation_deadline_stops &&
         left.materialization_deadline_stops ==
             right.materialization_deadline_stops &&
         left.generation_queue_drops == right.generation_queue_drops &&
         left.generation_retention_drops ==
             right.generation_retention_drops &&
         left.generation_boundary_replacements ==
             right.generation_boundary_replacements &&
         left.generation_tactical_shortcuts ==
             right.generation_tactical_shortcuts &&
         left.generation_fallbacks == right.generation_fallbacks &&
         left.generation_frontier_resumptions ==
             right.generation_frontier_resumptions &&
         left.generation_zero_action_resumptions ==
             right.generation_zero_action_resumptions &&
         left.generation_max_frontier_depth ==
             right.generation_max_frontier_depth &&
         left.progressive_widenings == right.progressive_widenings &&
         left.closed_unsolved_nodes == right.closed_unsolved_nodes &&
         left.closed_unsolved_nonexhaustive_nodes ==
             right.closed_unsolved_nonexhaustive_nodes &&
         left.open_unexpanded_nodes == right.open_unexpanded_nodes &&
         left.implicit_action_frontiers ==
             right.implicit_action_frontiers &&
         left.max_open_children == right.max_open_children &&
         left.tree_nodes == right.tree_nodes &&
         left.max_complete_turn_depth == right.max_complete_turn_depth &&
         left.root_value == right.root_value &&
         left.root_solved == right.root_solved &&
         left.proven_winner == right.proven_winner &&
         left.deadline_reached == right.deadline_reached &&
         left.tree_cap_reached == right.tree_cap_reached &&
         left.termination == right.termination &&
         left.cached_continuation == right.cached_continuation &&
         left.planned_action_length == right.planned_action_length &&
         left.current_edge_index == right.current_edge_index &&
         left.cached_moves_remaining == right.cached_moves_remaining &&
         left.searches == right.searches;
}

bool valid_search_completion(const ps::JacekReplayBfmSearchStats &stats,
                             std::size_t fixed_work_cap) {
  if (stats.deadline_reached || stats.closed_unsolved_nodes != 0U ||
      stats.closed_unsolved_nonexhaustive_nodes != 0U) {
    return false;
  }
  if (stats.root_solved) {
    return stats.proven_winner.has_value() &&
           stats.termination ==
               ps::JacekReplayBfmSearchTermination::RootSolved;
  }
  return !stats.proven_winner.has_value() && stats.tree_cap_reached &&
         stats.tree_nodes == fixed_work_cap &&
         stats.termination ==
             ps::JacekReplayBfmSearchTermination::FixedWorkCap;
}

std::string feature_indices_sha256(
    const ps::detail::ReplayBfmSparseFeatures &features) {
  std::vector<std::uint8_t> bytes(features.count * 2U);
  for (std::size_t index = 0; index < features.count; ++index) {
    bytes[index * 2U] =
        static_cast<std::uint8_t>(features.indices[index]);
    bytes[index * 2U + 1U] =
        static_cast<std::uint8_t>(features.indices[index] >> 8U);
  }
  return ps::detail::replay_bfm_sha256_hex(bytes);
}

void feature_schema_has_exact_categories_and_rotation() {
  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::detail::ReplayBfmSparseFeatures features =
      ps::detail::replay_bfm_sparse_features(initial);
  require(ps::detail::kReplayBfmInputCount == 6301U &&
              features.count == 105U,
          "Initial encoding must contain one active category per vertex.");
  require(feature_indices_sha256(features) ==
              "1ee1c7cf987fb1ca8d2bc707c4511b93fdf96216e6cf2a7917a18da6a2bec4d4",
          "The native initial features must match the Python exporter golden.");
  const ps::GameState after_north =
      ps::apply_move(initial, ps::Move{{4, 5}});
  const auto after_north_features =
      ps::detail::replay_bfm_sparse_features(after_north);
  require(after_north_features.count == 106U &&
              feature_indices_sha256(after_north_features) ==
                  "4b39d1752979d5c819cd8d4c4a022f56a0a0eb3227bd658c9fadf190dedf719f",
          "Player Two mover-relative features must match the Python golden.");
  const std::size_t ball_vertex = 5U * 9U + 4U;
  const std::uint16_t ball_input = static_cast<std::uint16_t>(
      316U + ball_vertex * 57U + 7U);
  require(std::find(features.indices.begin(),
                    features.indices.begin() + features.count,
                    ball_input) != features.indices.begin() + features.count,
          "The initial ball must encode distance zero and free degree eight.");

  const ps::Point corner{0, 1};
  for (const ps::Point neighbour :
       ps::neighbors(initial.config, corner, ps::Player::One)) {
    const ps::Segment edge{corner, neighbour};
    if (!ps::is_forbidden_boundary_segment(initial.config, edge)) {
      initial.used_segments.insert(edge);
    }
  }
  const auto isolated = ps::detail::replay_bfm_sparse_features(initial);
  const std::uint16_t unreachable = 316U + 56U;
  require(std::find(isolated.indices.begin(),
                    isolated.indices.begin() + isolated.count,
                    unreachable) != isolated.indices.begin() + isolated.count,
          "Distance seven or unreachable must use category 56.");

  ps::GameState asymmetric = ps::make_initial_state(codingame_rules());
  asymmetric.ball = ps::Point{5, 5};
  asymmetric.path = {{4, 6}, {5, 5}};
  asymmetric.visit_count = {{{4, 6}, 1}, {{5, 5}, 1}, {{6, 4}, 1}};
  asymmetric.used_segments = {
      ps::Segment{{4, 6}, {5, 5}}, ps::Segment{{6, 4}, {7, 3}}};
  require(ps::detail::replay_bfm_sparse_features(asymmetric) ==
              ps::detail::replay_bfm_sparse_features(
                  rotated_state(asymmetric)),
          "Player Two encoding must be exactly the 180-degree mover-relative rotation.");
}

void loader_and_float_inference_match_the_binary_contract() {
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const auto features = ps::detail::replay_bfm_sparse_features(state);
  const std::size_t input = features.indices.front();
  const std::size_t w1 = input * ps::detail::kReplayBfmHiddenOne;
  const std::size_t w2 = ps::detail::kReplayBfmInputCount *
                         ps::detail::kReplayBfmHiddenOne;
  const std::size_t w3 =
      w2 + ps::detail::kReplayBfmHiddenOne *
               ps::detail::kReplayBfmHiddenTwo;
  TemporaryCheckpoint checkpoint(
      make_checkpoint({{w1, 2.0F}, {w2, 3.0F}, {w3, 0.5F}}));
  const ps::detail::ReplayBfmModel model(checkpoint.string());
  require(std::abs(model.evaluate(features) - std::tanh(6.0F)) < 1e-6F,
          "Input-major sparse inference must apply square/leaky, leaky-ReLU, and tanh.");
  require(model.model_sha256().size() == 64U &&
              model.payload_sha256().size() == 64U &&
              ps::JacekReplayBfmBot::feature_schema_sha256() ==
                  ps::detail::kReplayBfmFeatureSchemaSha256,
          "The runtime and feature schema must expose stable SHA-256 identities.");
}

void prepared_first_layer_matches_full_sparse_inference() {
  ps::detail::ReplayBfmSparseFeatures before;
  before.indices = {3U, 17U, 316U, 987U};
  before.count = 4U;
  ps::detail::ReplayBfmSparseFeatures after;
  after.indices = {3U, 22U, 316U, 1044U, 6200U};
  after.count = 5U;

  const std::size_t w2 = ps::detail::kReplayBfmInputCount *
                         ps::detail::kReplayBfmHiddenOne;
  const std::size_t w3 =
      w2 + ps::detail::kReplayBfmHiddenOne *
               ps::detail::kReplayBfmHiddenTwo;
  TemporaryCheckpoint checkpoint(make_checkpoint({
      {3U * ps::detail::kReplayBfmHiddenOne, 0.1F},
      {17U * ps::detail::kReplayBfmHiddenOne, -0.2F},
      {22U * ps::detail::kReplayBfmHiddenOne, 0.3F},
      {316U * ps::detail::kReplayBfmHiddenOne, 0.4F},
      {987U * ps::detail::kReplayBfmHiddenOne, -0.5F},
      {1044U * ps::detail::kReplayBfmHiddenOne, 0.6F},
      {6200U * ps::detail::kReplayBfmHiddenOne, 0.7F},
      {w2, 0.8F},
      {w3, 0.9F},
  }));
  const ps::detail::ReplayBfmModel model(checkpoint.string());
  const float full = model.evaluate(after);
  const float delta = model.evaluate_delta(model.prepare(before), after);
  require(std::abs(full - delta) < 1e-6F,
          "Sibling evaluation deltas must reproduce full sparse inference.");
}

void loader_rejects_corruption_nonfinite_weights_and_trailing_bytes() {
  std::vector<std::uint8_t> corrupt = make_checkpoint();
  corrupt.back() ^= 1U;
  TemporaryCheckpoint bad_hash(corrupt);
  require_invalid(
      [&] { ps::detail::ReplayBfmModel model(bad_hash.string()); },
      "A stale payload hash must be rejected.");

  std::vector<std::uint8_t> nonfinite = make_checkpoint({
      {0U, std::numeric_limits<float>::quiet_NaN()},
  });
  TemporaryCheckpoint bad_weight(nonfinite);
  require_invalid(
      [&] { ps::detail::ReplayBfmModel model(bad_weight.string()); },
      "A hash-valid non-finite weight must be rejected.");

  std::vector<std::uint8_t> trailing = make_checkpoint();
  trailing.push_back(0U);
  TemporaryCheckpoint bad_size(trailing);
  require_invalid(
      [&] { ps::detail::ReplayBfmModel model(bad_size.string()); },
      "Trailing checkpoint bytes must be rejected.");

  std::vector<std::uint8_t> wrong_dimension = make_checkpoint();
  write_u32(wrong_dimension, 16U,
            static_cast<std::uint32_t>(ps::detail::kReplayBfmInputCount - 1U));
  TemporaryCheckpoint bad_dimension(wrong_dimension);
  require_invalid(
      [&] { ps::detail::ReplayBfmModel model(bad_dimension.string()); },
      "A mismatched network shape must be rejected.");

  std::vector<std::uint8_t> wrong_schema = make_checkpoint();
  wrong_schema[56U] ^= 1U;
  TemporaryCheckpoint bad_schema(wrong_schema);
  require_invalid(
      [&] { ps::detail::ReplayBfmModel model(bad_schema.string()); },
      "A mismatched feature schema must be rejected.");

  std::vector<std::uint8_t> reserved = make_checkpoint();
  reserved[127U] = 1U;
  TemporaryCheckpoint bad_reserved(reserved);
  require_invalid(
      [&] { ps::detail::ReplayBfmModel model(bad_reserved.string()); },
      "Nonzero reserved header bytes must be rejected.");
}

void checked_in_bootstrap_matches_the_native_loader() {
  const std::string source_dir = PAPERSOCCER_SOURCE_DIR;
  if (source_dir.empty()) {
    return;
  }
  const std::filesystem::path path =
      std::filesystem::path(source_dir) /
      "models/jacek_replay_bfm_bootstrap.runtime";
  const ps::detail::ReplayBfmModel model(path.string());
  const float initial_prediction = model.evaluate(
      ps::detail::replay_bfm_sparse_features(
          ps::make_initial_state(codingame_rules())));
  require(model.model_sha256() ==
              "02c7757443285cf6d10e971f25dee0869339d76e1d09bb9616ce5b83966cabf7" &&
              model.payload_sha256() ==
                  "71856d2c3ae577ba83ae8efc6ebbbd1b574d0a7c207651148febe4e9257d08b8" &&
              std::isfinite(initial_prediction) &&
              std::abs(initial_prediction - 0.000798130699F) < 1e-6F,
          "The checked-in bootstrap must match exporter hashes and load natively.");
}

void checked_in_development_candidate_matches_the_native_loader() {
  const std::string source_dir = PAPERSOCCER_SOURCE_DIR;
  if (source_dir.empty()) {
    return;
  }
  const std::filesystem::path path =
      std::filesystem::path(source_dir) / "models" /
      "jacek_replay_bfm_development" / "jacek_replay_bfm.runtime";
  const ps::detail::ReplayBfmModel model(path.string());
  const float initial_prediction = model.evaluate(
      ps::detail::replay_bfm_sparse_features(
          ps::make_initial_state(codingame_rules())));
  require(model.model_sha256() ==
              "7e50b4f9b0c407c1e649426422f498efb0618be808e0f89a2c48548ad21844fa" &&
              std::isfinite(initial_prediction),
          "The checked-in development candidate must retain its reviewed identity.");
}

ps::GameState forced_rebound_state() {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  const ps::Point rebound{4, 5};
  state.visit_count[rebound] = 1;
  for (const ps::Move move : ps::legal_moves(state)) {
    if (move.to != rebound) {
      state.used_segments.insert(ps::Segment{state.ball, move.to});
    }
  }
  return state;
}

void block_point_except(ps::GameState &state, ps::Point point,
                        const std::vector<ps::Point> &allowed) {
  for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
    for (const ps::Point destination :
         ps::neighbors(state.config, point, player)) {
      if (std::find(allowed.begin(), allowed.end(), destination) ==
              allowed.end() &&
          !ps::is_forbidden_boundary_segment(
              state.config, ps::Segment{point, destination})) {
        state.used_segments.insert(ps::Segment{point, destination});
      }
    }
  }
}

void bot_searches_complete_turns_and_reuses_rebound_edges() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.seed = 17U;
  config.max_time_ms = 100U;
  config.max_tree_nodes = 2U;
  config.max_actions = 1U;
  config.max_partial_paths = 1'000U;
  ps::JacekReplayBfmBot bot(config);

  ps::GameState state = forced_rebound_state();
  const ps::Move first = bot.choose_move(state);
  require(first == ps::Move{{4, 5}} &&
              bot.last_search_stats().planned_action_length >= 2U &&
              bot.last_search_stats().cached_moves_remaining >= 1U &&
              bot.last_search_stats().tree_nodes == 2U,
          "The BFM must plan a complete forced rebound action.");
  state = ps::apply_move(state, first);
  const ps::Move second = bot.choose_move(state);
  const std::vector<ps::Move> legal = ps::legal_moves(state);
  require(std::find(legal.begin(), legal.end(), second) != legal.end() &&
              bot.last_search_stats().cached_continuation &&
              bot.last_search_stats().current_edge_index == 1U,
          "The next rebound edge must come from the validated action cache.");

  ps::JacekReplayBfmBot invalidated(config);
  ps::GameState mismatched = forced_rebound_state();
  const ps::Move initial = invalidated.choose_move(mismatched);
  mismatched = ps::apply_move(mismatched, initial);
  mismatched.path.push_back(mismatched.ball);
  const ps::Move replacement = invalidated.choose_move(mismatched);
  const std::vector<ps::Move> replacement_legal = ps::legal_moves(mismatched);
  require(std::find(replacement_legal.begin(), replacement_legal.end(),
                    replacement) != replacement_legal.end() &&
              !invalidated.last_search_stats().cached_continuation &&
              invalidated.last_search_stats().searches == 2U &&
              invalidated.last_search_stats().neural_evaluations > 0U,
          "Any expected-state mismatch must discard the rebound cache and search.");
}

void complete_turn_generation_stops_at_the_action_cap() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.seed = 0x243f6a8885a308d3ULL;
  config.max_time_ms = 1'000U;
  config.max_tree_nodes = 3U;
  config.max_actions = 2U;
  config.max_partial_paths = 50'000U;
  ps::JacekReplayBfmBot bot(config);

  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const std::vector<ps::Move> legal = ps::legal_moves(state);
  const ps::Move move = bot.choose_move(state);
  const ps::JacekReplayBfmSearchStats &stats = bot.last_search_stats();
  require(std::find(legal.begin(), legal.end(), move) != legal.end() &&
              stats.retained_actions == 2U && stats.partial_paths < 100U &&
              stats.truncations > 0U && !stats.deadline_reached,
          "A full retained-action sample must stop complete-turn enumeration.");
}

void tactical_goal_witnesses_win_for_both_players() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.max_time_ms = 100U;
  config.max_tree_nodes = 2U;
  config.max_actions = 1U;
  config.max_partial_paths = 64U;
  ps::JacekReplayBfmBot bot(config);

  ps::GameState one = ps::make_initial_state(codingame_rules());
  one.ball = {4, 1};
  one.path = {one.ball};
  one.visit_count = {{one.ball, 1}};
  const ps::Move one_goal = bot.choose_move(one);
  require(ps::winner(ps::apply_move(one, one_goal)) == ps::Player::One &&
              bot.last_search_stats().tactical_solutions > 0U &&
              bot.last_search_stats().partial_paths == 0U,
          "Player One must retain and select an immediate north goal proof.");

  ps::GameState two = ps::make_initial_state(codingame_rules());
  two.ball = {4, 11};
  two.to_move = ps::Player::Two;
  two.path = {two.ball};
  two.visit_count = {{two.ball, 1}};
  ps::GameState played = two;
  const ps::Move first_two_edge = bot.choose_move(played);
  const bool retained_proof =
      bot.last_search_stats().tactical_solutions > 0U;
  played = ps::apply_move(played, first_two_edge);
  std::size_t edges = 1U;
  while (!ps::is_terminal(played) && played.to_move == ps::Player::Two) {
    const ps::Move continuation = bot.choose_move(played);
    require(bot.last_search_stats().cached_continuation,
            "A retained multi-edge goal proof must use the rebound cache.");
    played = ps::apply_move(played, continuation);
    require(++edges <= 64U, "The retained Player Two goal proof must terminate.");
  }
  require(ps::winner(played) == ps::Player::Two && retained_proof,
          "Player Two must retain and complete a same-turn south goal proof.");
}

void tactical_handoff_proofs_have_the_correct_minimax_sign() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.seed = 29U;
  config.max_time_ms = 100U;
  config.max_tree_nodes = 2U;
  config.max_actions = 1U;
  config.max_partial_paths = 128U;

  ps::GameState cutoff = ps::make_initial_state(codingame_rules());
  cutoff.ball = {4, 6};
  cutoff.path = {cutoff.ball};
  cutoff.visit_count = {{{4, 6}, 1}, {{4, 4}, 1}};
  block_point_except(cutoff, {4, 6}, {{4, 5}});
  block_point_except(cutoff, {4, 5}, {{4, 6}, {4, 4}});
  block_point_except(cutoff, {4, 4}, {{4, 5}});
  ps::JacekReplayBfmBot cutoff_bot(config);
  const ps::Move cutoff_move = cutoff_bot.choose_move(cutoff);
  const ps::GameState cutoff_child = ps::apply_move(cutoff, cutoff_move);
  require(cutoff_move == ps::Move{{4, 5}} &&
              !ps::is_terminal(cutoff_child) &&
              cutoff_child.to_move == ps::Player::Two &&
              cutoff_bot.last_search_stats().tactical_solutions > 0U &&
              cutoff_bot.last_search_stats().partial_paths == 0U &&
              cutoff_bot.last_search_stats().root_value > 1'000.0F,
          "A handoff that forces the opponent to be blocked must back up as a win.");

  ps::GameState immediate_reply = ps::make_initial_state(codingame_rules());
  immediate_reply.ball = {4, 10};
  immediate_reply.path = {immediate_reply.ball};
  immediate_reply.visit_count = {{{4, 10}, 1}};
  block_point_except(immediate_reply, {4, 10}, {{4, 11}});
  ps::JacekReplayBfmBot reply_bot(config);
  const ps::Move reply_move = reply_bot.choose_move(immediate_reply);
  const ps::GameState reply_child = ps::apply_move(immediate_reply, reply_move);
  require(reply_move == ps::Move{{4, 11}} &&
              !ps::is_terminal(reply_child) &&
              reply_child.to_move == ps::Player::Two &&
              reply_bot.last_search_stats().tactical_solutions > 0U &&
              reply_bot.last_search_stats().root_value < -1'000.0F,
          "A handoff that gives the opponent an immediate goal must back up as a loss.");
}

void universal_loss_requires_resumed_frontier_exhaustion() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.max_time_ms = 1'000U;
  config.max_tree_nodes = 3U;
  config.max_actions = 1U;
  config.max_partial_paths = 64U;

  ps::GameState state = ps::make_initial_state(codingame_rules());
  state.ball = {4, 11};
  state.path = {state.ball};
  state.visit_count = {{state.ball, 1}};
  block_point_except(state, state.ball, {{3, 12}, {5, 12}});
  ps::JacekReplayBfmBot bot(config);
  const ps::JacekReplayBfmSearchStats stats =
      bot.analyze_position(state, 41U);
  require(stats.root_solved &&
              stats.proven_winner == ps::Player::Two &&
              stats.termination ==
                  ps::JacekReplayBfmSearchTermination::RootSolved &&
              stats.progressive_widenings > 0U &&
              stats.generation_frontier_resumptions > 0U &&
              stats.tree_nodes == 3U,
          "A universal loss must enumerate every paged action before proof.");
}

void fixed_cap_search_is_mover_rotation_symmetric() {
  ps::GameState player_one = ps::make_initial_state(codingame_rules());
  player_one = ps::apply_move(player_one, ps::Move{{4, 5}});
  player_one = ps::apply_move(player_one, ps::Move{{5, 4}});
  require(player_one.to_move == ps::Player::One,
          "The symmetry fixture must be a Player One boundary.");
  ps::GameState player_two = rotated_state(player_one);

  const auto active = ps::detail::replay_bfm_sparse_features(player_one);
  const std::size_t w1 =
      active.indices.front() * ps::detail::kReplayBfmHiddenOne;
  const std::size_t w2 = ps::detail::kReplayBfmInputCount *
                         ps::detail::kReplayBfmHiddenOne;
  const std::size_t w3 =
      w2 + ps::detail::kReplayBfmHiddenOne *
               ps::detail::kReplayBfmHiddenTwo;
  TemporaryCheckpoint checkpoint(
      make_checkpoint({{w1, 0.25F}, {w2, 0.5F}, {w3, 0.5F}}));
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.seed = 0x6a09e667f3bcc909ULL;
  config.max_time_ms = 1'000U;
  config.max_tree_nodes = 47U;
  config.max_actions = 5U;
  config.max_partial_paths = 11U;

  ps::JacekReplayBfmBot one(config);
  ps::JacekReplayBfmBot two(config);
  const ps::Move first_one = one.choose_move(player_one);
  const ps::JacekReplayBfmSearchStats one_stats = one.last_search_stats();
  const ps::Move first_two = two.choose_move(player_two);
  const ps::JacekReplayBfmSearchStats two_stats = two.last_search_stats();
  require(first_two.to == rotate(first_one.to),
          "A capped Player Two root action must rotate from Player One.");
  require(one_stats.tree_cap_reached && two_stats.tree_cap_reached &&
              std::abs(one_stats.root_value) > 1e-6F &&
              one_stats.root_value == two_stats.root_value &&
              same_capped_search_stats(one_stats, two_stats),
          "Fixed-cap rotated searches must have identical stats and value.");

  ps::JacekReplayBfmBot repeat(config);
  const ps::Move repeated_move = repeat.choose_move(player_one);
  require(repeated_move == first_one &&
              same_capped_search_stats(repeat.last_search_stats(), one_stats),
          "The same seed and fixed work cap must reproduce move and stats exactly.");

  player_one = ps::apply_move(player_one, first_one);
  player_two = ps::apply_move(player_two, first_two);
  std::size_t edge = 1U;
  while (!ps::is_terminal(player_one) &&
         player_one.to_move == ps::Player::One) {
    require(!ps::is_terminal(player_two) &&
                player_two.to_move == ps::Player::Two,
            "Rotated complete actions must hand off on the same edge.");
    const ps::Move next_one = one.choose_move(player_one);
    const ps::Move next_two = two.choose_move(player_two);
    require(next_two.to == rotate(next_one.to) &&
                one.last_search_stats().cached_continuation &&
                two.last_search_stats().cached_continuation &&
                same_capped_search_stats(one.last_search_stats(),
                                         two.last_search_stats()),
            "Every cached edge of the capped action must rotate exactly.");
    player_one = ps::apply_move(player_one, next_one);
    player_two = ps::apply_move(player_two, next_two);
    require(++edge <= one_stats.planned_action_length,
            "The symmetry action exceeded its planned length.");
  }
  require(ps::is_terminal(player_one) == ps::is_terminal(player_two) &&
              player_two.ball == rotate(player_one.ball) &&
              player_two.to_move == ps::opponent(player_one.to_move),
          "Rotated capped actions must finish at equivalent boundaries.");
}

void deadline_fallback_is_legal_and_bounded() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.max_time_ms = 1U;
  config.max_tree_nodes = 1'000'000U;
  config.max_actions = 250U;
  config.max_partial_paths = 50'000U;
  ps::JacekReplayBfmBot bot(config);
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const auto start = std::chrono::steady_clock::now();
  const ps::Move move = bot.choose_move(state);
  const auto elapsed = std::chrono::steady_clock::now() - start;
  const std::vector<ps::Move> legal = ps::legal_moves(state);
  require(std::find(legal.begin(), legal.end(), move) != legal.end() &&
              bot.last_search_stats().deadline_reached &&
              elapsed < std::chrono::milliseconds(500),
          "A one-millisecond deadline must return a prompt legal fallback.");
}

void zero_time_disables_the_internal_deadline() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.max_time_ms = 0U;
  config.max_tree_nodes = 47U;
  config.max_actions = 5U;
  config.max_partial_paths = 11U;
  config.exploration = 0.5;
  config.fpu = 0.5;
  ps::JacekReplayBfmBot bot(config);
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const ps::JacekReplayBfmSearchStats first =
      bot.analyze_position(state, 31U);
  const ps::JacekReplayBfmSearchStats repeated =
      bot.analyze_position(state, 31U);
  require(!first.deadline_reached &&
              first.generation_deadline_stops == 0U &&
              first.materialization_deadline_stops == 0U &&
              valid_search_completion(first, config.max_tree_nodes) &&
              same_capped_search_stats(first, repeated),
          "A zero time budget must disable wall-clock termination while "
          "preserving deterministic proof-or-fixed-cap completion.");
}

void analysis_api_is_stateless_seeded_rotatable_and_proof_explicit() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::JacekReplayBfmConfig config;
  config.model_path = checkpoint.string();
  config.max_time_ms = 1'000U;
  config.max_tree_nodes = 47U;
  config.max_actions = 5U;
  config.max_partial_paths = 11U;
  config.exploration = 0.5;
  config.fpu = 0.5;
  ps::JacekReplayBfmBot bot(config);

  ps::GameState player_one = ps::make_initial_state(codingame_rules());
  player_one = ps::apply_move(player_one, ps::Move{{4, 5}});
  player_one = ps::apply_move(player_one, ps::Move{{5, 4}});
  const ps::GameState player_two = rotated_state(player_one);
  const std::uint64_t seed = 0xbb67ae8584caa73bULL;
  const ps::JacekReplayBfmSearchStats first =
      bot.analyze_position(player_one, seed);
  const ps::JacekReplayBfmSearchStats repeated =
      bot.analyze_position(player_one, seed);
  const ps::JacekReplayBfmSearchStats rotated =
      bot.analyze_position(player_two, seed);
  require(first.tree_cap_reached && !first.root_solved &&
              !first.proven_winner.has_value() &&
              same_capped_search_stats(first, repeated) &&
              same_capped_search_stats(first, rotated) &&
              bot.last_search_stats().tree_nodes == 0U,
          "Stateless analysis must reuse one model, honor its per-position "
          "seed, rotate exactly, and leave move-cache diagnostics untouched.");

  ps::JacekReplayBfmConfig proof_config = config;
  proof_config.max_tree_nodes = 2U;
  proof_config.max_actions = 1U;
  proof_config.max_partial_paths = 128U;
  ps::JacekReplayBfmBot proof_bot(proof_config);
  ps::GameState cutoff = ps::make_initial_state(codingame_rules());
  cutoff.ball = {4, 6};
  cutoff.path = {cutoff.ball};
  cutoff.visit_count = {{{4, 6}, 1}, {{4, 4}, 1}};
  block_point_except(cutoff, {4, 6}, {{4, 5}});
  block_point_except(cutoff, {4, 5}, {{4, 6}, {4, 4}});
  block_point_except(cutoff, {4, 4}, {{4, 5}});
  const ps::JacekReplayBfmSearchStats one_proof =
      proof_bot.analyze_position(cutoff, 29U);
  const ps::JacekReplayBfmSearchStats two_proof =
      proof_bot.analyze_position(rotated_state(cutoff), 29U);
  require(one_proof.root_solved &&
              one_proof.proven_winner == ps::Player::One &&
              one_proof.root_value > 1'000.0F && two_proof.root_solved &&
              two_proof.proven_winner == ps::Player::Two &&
              two_proof.root_value == one_proof.root_value,
          "Solved roots must expose an explicit absolute winner rather than "
          "requiring callers to infer a proof from the value magnitude.");
}

void nonexhaustive_root_progressively_widens_to_a_valid_label() {
  const std::string source_dir = PAPERSOCCER_SOURCE_DIR;
  if (source_dir.empty()) {
    return;
  }
  const std::filesystem::path model =
      std::filesystem::path(source_dir) / "models" /
      "jacek_replay_bfm_development" / "jacek_replay_bfm.runtime";
  ps::JacekReplayBfmConfig config;
  config.model_path = model.string();
  config.max_time_ms = 60'000U;
  config.max_tree_nodes = 64'000U;
  config.max_actions = 250U;
  config.max_partial_paths = 50'000U;
  config.exploration = 0.5;
  config.fpu = 0.5;
  ps::JacekReplayBfmBot bot(config);

  // Preserved campaign position
  // position:fc5350b1be0e9887a10b97bd5569a1251a9824cb4a4e9c6d7404471eeb4616e3
  const ps::GameState state = replay_complete_turn_prefix(
      "1/2/7/5/207/6/1/45/00/75/03/35/22/445/7/2/177/44/47/"
      "2345/7/53/0/23/0/3/1/4/1/3/17/54/1/36350/00017/25/017/27/"
      "035075/5433/00/35235663357/025766/752530/010/245021/"
      "065052507117/723064534/17/7245201235/05223/63/00117");
  require(!ps::is_terminal(state) && state.to_move == ps::Player::Two,
          "The preserved regression prefix must end at a Player Two root.");

  constexpr std::uint64_t seed = 11'313'244'602'099'372'890ULL;
  const ps::JacekReplayBfmSearchStats first =
      bot.analyze_position(state, seed);
  const ps::JacekReplayBfmSearchStats repeated =
      bot.analyze_position(state, seed);
  require(valid_search_completion(first, config.max_tree_nodes) &&
              first.progressive_widenings > 0U &&
              first.generation_frontier_resumptions > 0U &&
              first.max_open_children <= config.max_actions &&
              same_capped_search_stats(first, repeated),
          "A non-exhaustive root whose sampled actions all close must widen "
          "deterministically until it has an explicit proof or fixed-work cap.");

  ps::JacekReplayBfmConfig narrow_config = config;
  narrow_config.max_tree_nodes = 200U;
  narrow_config.max_actions = 1U;
  ps::JacekReplayBfmBot narrow(narrow_config);
  constexpr std::uint64_t cross_cap_seed =
      11'703'771'591'024'012'692ULL;
  const ps::JacekReplayBfmSearchStats narrow_stats =
      narrow.analyze_position(state, cross_cap_seed);
  require(valid_search_completion(narrow_stats,
                                  narrow_config.max_tree_nodes),
          "A truncated one-action page must not become an exhaustive proof.");

  ps::JacekReplayBfmConfig stalled_config = narrow_config;
  stalled_config.max_partial_paths = 1U;
  ps::JacekReplayBfmBot stalled(stalled_config);
  const ps::JacekReplayBfmSearchStats stalled_stats =
      stalled.analyze_position(state, cross_cap_seed);
  require(valid_search_completion(stalled_stats,
                                  stalled_config.max_tree_nodes) &&
              stalled_stats.generation_partial_cap_stops > 0U &&
              stalled_stats.generation_frontier_resumptions > 0U,
          "A one-partial cursor must resume without inventing fixed work or "
          "waiting for a deadline.");

  // Production v2 failure position.  The checked-in development model does
  // not stall at its production limits, so the small page deliberately
  // exercises the same cursor-resume mechanism with portable model bytes.
  const ps::GameState production_failure = replay_complete_turn_prefix(
      "2/1/4/1/7/4217/6/41174/36056/5/4/7/0/366/11/75/36/0752/345/71/"
      "6113/5602534/7/4217553/13/6/7524/53/1/4/6530/12/1/6/47716524/"
      "313/01/0254/2/5/052723/65/06331/002/7/75/72330660050/643/"
      "2505653001/1430311");
  require(!ps::is_terminal(production_failure) &&
              production_failure.to_move == ps::Player::One,
          "The production frontier regression must end at a Player One root.");
  ps::JacekReplayBfmConfig production_config = config;
  production_config.max_tree_nodes = 200U;
  production_config.max_actions = 1U;
  production_config.max_partial_paths = 1U;
  ps::JacekReplayBfmBot production_bot(production_config);
  constexpr std::uint64_t production_seed =
      11'914'096'561'988'050'746ULL;
  const ps::JacekReplayBfmSearchStats production_stats =
      production_bot.analyze_position(production_failure, production_seed);
  const ps::JacekReplayBfmSearchStats production_repeat =
      production_bot.analyze_position(production_failure, production_seed);
  require(valid_search_completion(production_stats,
                                  production_config.max_tree_nodes) &&
              production_stats.generation_partial_cap_stops > 0U &&
              production_stats.generation_frontier_resumptions > 0U &&
              production_stats.generation_zero_action_resumptions > 0U &&
              production_stats.generation_max_frontier_depth > 0U &&
              production_stats.generation_queue_drops == 0U &&
              same_capped_search_stats(production_stats, production_repeat),
          "The production partial-frontier failure must resume deterministically "
          "until proof or the exact fixed-work cap.");
}

void public_factory_and_rule_contract_fail_closed() {
  TemporaryCheckpoint checkpoint(make_checkpoint());
  ps::BotConfig factory;
  factory.kind = ps::BotKind::JacekReplayBfm;
  factory.jacek_replay_bfm.model_path = checkpoint.string();
  factory.jacek_replay_bfm.max_time_ms = 10U;
  factory.jacek_replay_bfm.max_tree_nodes = 2U;
  factory.jacek_replay_bfm.max_actions = 1U;
  factory.jacek_replay_bfm.max_partial_paths = 8U;
  std::unique_ptr<ps::Bot> base = ps::make_bot(factory);
  const auto *bot = dynamic_cast<const ps::JacekReplayBfmBot *>(base.get());
  require(bot != nullptr && base->name() == "JacekReplayBfmBot" &&
              ps::bot_kind_name(ps::BotKind::JacekReplayBfm) ==
                  "JacekReplayBfmBot" &&
              bot->config().model_path == checkpoint.string(),
          "The public factory must preserve the new bot configuration.");

  require_invalid(
      [&] {
        ps::JacekReplayBfmConfig missing;
        ps::JacekReplayBfmBot invalid(missing);
      },
      "The external model path must be required.");
  require_invalid(
      [&] {
        ps::GameState wrong = ps::make_initial_state();
        (void)base->choose_move(wrong);
      },
      "Rules outside OwnGoalsAllowed plus MoverLoses must be rejected.");
}

}  // namespace

int run_jacek_replay_bfm_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"feature_schema_has_exact_categories_and_rotation",
       feature_schema_has_exact_categories_and_rotation},
      {"loader_and_float_inference_match_the_binary_contract",
       loader_and_float_inference_match_the_binary_contract},
      {"prepared_first_layer_matches_full_sparse_inference",
       prepared_first_layer_matches_full_sparse_inference},
      {"loader_rejects_corruption_nonfinite_weights_and_trailing_bytes",
       loader_rejects_corruption_nonfinite_weights_and_trailing_bytes},
      {"checked_in_bootstrap_matches_the_native_loader",
       checked_in_bootstrap_matches_the_native_loader},
      {"checked_in_development_candidate_matches_the_native_loader",
       checked_in_development_candidate_matches_the_native_loader},
      {"bot_searches_complete_turns_and_reuses_rebound_edges",
       bot_searches_complete_turns_and_reuses_rebound_edges},
      {"complete_turn_generation_stops_at_the_action_cap",
       complete_turn_generation_stops_at_the_action_cap},
      {"tactical_goal_witnesses_win_for_both_players",
       tactical_goal_witnesses_win_for_both_players},
      {"tactical_handoff_proofs_have_the_correct_minimax_sign",
       tactical_handoff_proofs_have_the_correct_minimax_sign},
      {"universal_loss_requires_resumed_frontier_exhaustion",
       universal_loss_requires_resumed_frontier_exhaustion},
      {"fixed_cap_search_is_mover_rotation_symmetric",
       fixed_cap_search_is_mover_rotation_symmetric},
      {"deadline_fallback_is_legal_and_bounded",
       deadline_fallback_is_legal_and_bounded},
      {"zero_time_disables_the_internal_deadline",
       zero_time_disables_the_internal_deadline},
      {"analysis_api_is_stateless_seeded_rotatable_and_proof_explicit",
       analysis_api_is_stateless_seeded_rotatable_and_proof_explicit},
      {"nonexhaustive_root_progressively_widens_to_a_valid_label",
       nonexhaustive_root_progressively_widens_to_a_valid_label},
      {"public_factory_and_rule_contract_fail_closed",
       public_factory_and_rule_contract_fail_closed},
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
            << '/' << tests.size() << " Jacek replay BFM tests passed.\n";
  return failures == 0 ? 0 : 1;
}
