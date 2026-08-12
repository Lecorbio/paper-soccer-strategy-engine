#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include "bot.cpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace candidate = papersoccer::jacek_native_bfm;
namespace model = papersoccer::jacek_native_model;

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

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::GameState make_clean_state_at(ps::Point point,
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

bool same_state(const ps::GameState &left, const ps::GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

void block_edges_except(ps::GameState &state, ps::Point from,
                        const std::vector<ps::Point> &allowed) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const ps::Point destination{from.x + dx, from.y + dy};
      if (std::find(allowed.begin(), allowed.end(), destination) ==
          allowed.end()) {
        state.used_segments.insert(ps::Segment{from, destination});
      }
    }
  }
}

ps::GameState forced_two_move_rebound() {
  const ps::Point root{4, 4};
  const ps::Point rebound{4, 3};
  const ps::Point handoff{4, 2};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, handoff});
  return state;
}

ps::GameState dense_rebound_state() {
  ps::GameState state = make_clean_state_at({4, 6});
  for (int x = 2; x <= 6; ++x) {
    for (int y = 4; y <= 8; ++y) {
      state.visit_count[{x, y}] = 1;
    }
  }
  return state;
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

ps::Point reflect(ps::Point point) {
  return ps::Point{8 - point.x, point.y};
}

ps::GameState reflected_state(const ps::GameState &state) {
  ps::GameState result;
  result.config = state.config;
  result.ball = reflect(state.ball);
  result.to_move = state.to_move;
  result.status = state.status;
  for (const ps::Point point : state.path) {
    result.path.push_back(reflect(point));
  }
  for (const ps::Segment &segment : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{reflect(segment.a), reflect(segment.b)});
  }
  for (const auto &[point, visits] : state.visit_count) {
    result.visit_count.emplace(reflect(point), visits);
  }
  return result;
}

char reflected_direction(char direction) {
  constexpr std::array<char, 8> reflected{{'0', '7', '6', '5',
                                           '4', '3', '2', '1'}};
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("invalid direction to reflect");
  }
  return reflected[static_cast<std::size_t>(direction - '0')];
}

std::string reflected_action(std::string action) {
  for (char &direction : action) {
    direction = reflected_direction(direction);
  }
  return action;
}

void constants_freeze_the_direct_contract() {
  static_assert(candidate::kMaximumActions == 250);
  static_assert(candidate::kProductionPartialPaths == 4'000);
  static_assert(candidate::kNonrootPartialPaths == 512);
  static_assert(candidate::kProductionTreeNodes == 80'000);
  static_assert(candidate::kFifoPeriod == 10);
  static_assert(candidate::kLifoExtractionsPerPeriod == 9);
  static_assert(candidate::kFeatureInputs == 1156);
  static_assert(candidate::kEdgeFeatureInputs == 316);
  static_assert(candidate::kFeatureVertices == 105);
  static_assert(candidate::kDistanceBuckets == 8);
  static_assert(candidate::SearchConfig{}.shuffle_seed ==
                0x6a09e667f3bcc909ULL);
  require(candidate::kExplorationConstant == 0.80,
          "The deployed UCT constant must remain explicit.");
  require(candidate::kFirstPlayUrgency == 0.5,
          "The deployed first-play urgency must remain explicit.");
}

void uct_fpu_and_final_visit_formulas_are_literal() {
  const double unvisited = candidate::uct_action_score(0.25F, 100, 0);
  require(std::abs(unvisited - 0.75) <= 1e-12,
          "Unvisited actions must add literal FPU 0.5.");
  const double expected_visited =
      0.25 + 0.80 * std::sqrt(std::log(100.0) / 4.0);
  require(std::abs(candidate::uct_action_score(0.25F, 100, 4) -
                   expected_visited) <= 1e-12,
          "Visited actions must use literal C=0.80 UCT.");
  require(std::abs(candidate::final_action_score(0.25F, 7) -
                   (0.25 + std::log(7.0))) <= 1e-12,
          "Final ordering must be value plus log(visits).");
  require_invalid_argument(
      [] { (void)candidate::final_action_score(0.0F, 0); },
      "An unvisited final action must be rejected.");
}

void directions_round_trip_exactly() {
  const ps::Point origin{4, 6};
  constexpr std::array<ps::Point, 8> expected{{
      {4, 5}, {5, 5}, {5, 6}, {5, 7},
      {4, 7}, {3, 7}, {3, 6}, {3, 5},
  }};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const char direction = static_cast<char>('0' + index);
    require(candidate::decode_direction(origin, direction).to == expected[index],
            "Direction decoding must match the CodinGame compass table.");
    require(candidate::encode_direction(origin, expected[index]) == direction,
            "Direction encoding must invert every valid direction.");
  }
  require_invalid_argument(
      [&] { (void)candidate::decode_direction(origin, '8'); },
      "Direction eight must never be accepted or emitted.");
}

void own_goals_and_blocked_mover_follow_contest_rules() {
  ps::GameState north = make_clean_state_at({4, 1}, ps::Player::Two);
  north = ps::apply_move(north, {{4, 0}});
  require(ps::winner(north) == ps::Player::One,
          "Player Two's north own goal must award Player One.");

  ps::GameState south = make_clean_state_at({4, 11}, ps::Player::One);
  south = ps::apply_move(south, {{4, 12}});
  require(ps::winner(south) == ps::Player::Two,
          "Player One's south own goal must award Player Two.");

  ps::GameState blocked = make_clean_state_at({4, 6}, ps::Player::One);
  block_edges_except(blocked, {4, 5}, {{4, 6}});
  blocked = ps::apply_move(blocked, {{4, 5}});
  require(ps::winner(blocked) == ps::Player::Two,
          "A blocked landing must immediately defeat the mover.");
}

void encoded_turn_validation_is_atomic() {
  ps::GameState rebound = forced_two_move_rebound();
  const ps::GameState snapshot = rebound;
  require_invalid_argument(
      [&] { candidate::apply_encoded_turn(rebound, "0"); },
      "A turn may not stop during a mandatory rebound.");
  require(same_state(rebound, snapshot),
          "A rejected early turn must not partially mutate state.");

  candidate::apply_encoded_turn(rebound, "00");
  require(rebound.ball == ps::Point{4, 2} &&
              rebound.to_move == ps::Player::Two,
          "A complete rebound sequence must hand possession over.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::GameState initial_snapshot = initial;
  require_invalid_argument(
      [&] { candidate::apply_encoded_turn(initial, "00"); },
      "A turn may not include an opponent's next move.");
  require(same_state(initial, initial_snapshot),
          "A rejected extra move must not partially mutate state.");
}

void initial_features_match_the_article_contract() {
  const candidate::DenseFeatures features =
      candidate::encode_features(ps::make_initial_state(codingame_rules()));
  require(features.size() == 1156,
          "The model input must contain exactly 1,156 features.");
  require(std::all_of(features.begin(), features.begin() + 316,
                      [](std::uint8_t value) { return value == 0; }),
          "The initial state must not activate a used-edge input.");

  std::array<std::size_t, 8> histogram{};
  for (std::size_t vertex = 0; vertex < 105; ++vertex) {
    std::size_t active = 0;
    for (std::size_t bucket = 0; bucket < 8; ++bucket) {
      const std::uint8_t value = features[316 + vertex * 8 + bucket];
      active += value;
      histogram[bucket] += value;
    }
    require(active == 1,
            "Every canonical vertex must have one distance bucket.");
  }
  require(histogram == std::array<std::size_t, 8>{1, 8, 16, 46, 28, 6, 0, 0},
          "Initial true-turn distances must preserve the reviewed ordering.");
}

void feature_order_and_player_rotation_are_bit_exact() {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  for (const ps::Point destination : {
           ps::Point{3, 5}, ps::Point{2, 5},
           ps::Point{3, 6}, ps::Point{4, 6},
       }) {
    state = ps::apply_move(state, ps::Move{destination});
  }
  require(state.to_move == ps::Player::Two &&
              state.used_segments.size() == 4,
          "The feature fixture must remain a Player Two rebound state.");
  const candidate::DenseFeatures features = candidate::encode_features(state);
  std::uint64_t hash = 0xcbf29ce484222325ULL;
  std::size_t active = 0;
  for (std::size_t index = 0; index < features.size(); ++index) {
    if (features[index] == 0) {
      continue;
    }
    require(features[index] == 1, "Article inputs must remain binary.");
    ++active;
    hash ^= index & 0xffU;
    hash *= 0x100000001b3ULL;
    hash ^= (index >> 8U) & 0xffU;
    hash *= 0x100000001b3ULL;
  }
  require(active == 109 && hash == 0xf90097cecb3bbbb8ULL,
          "The edge/vertex feature index order changed.");

  ps::GameState player_one = ps::make_initial_state(codingame_rules());
  player_one.ball = {3, 5};
  player_one.to_move = ps::Player::One;
  player_one.path = {{4, 6}, {3, 5}};
  player_one.used_segments = {
      ps::Segment{{4, 6}, {3, 5}},
      ps::Segment{{3, 5}, {2, 5}},
      ps::Segment{{2, 5}, {2, 4}},
  };
  player_one.visit_count.clear();
  player_one.visit_count[{4, 6}] = 1;
  player_one.visit_count[{3, 5}] = 2;
  player_one.visit_count[{2, 5}] = 1;
  player_one.visit_count[{2, 4}] = 1;
  require(candidate::encode_features(player_one) ==
              candidate::encode_features(rotated_state(player_one)),
          "Player Two's 180-degree canonicalization must be bit exact.");
}

void model_artifact_identity_is_frozen() {
  static_assert(model::kInputs == 1156);
  static_assert(model::kHiddenOne == 32);
  static_assert(model::kHiddenTwo == 32);
  static_assert(model::kOutputs == 1);
  static_assert(model::kWeightCount == 38'048);
  static_assert(model::kQuantizationBits == 3);
  static_assert(model::kTrainingSeed == 20260822ULL);
  require(model::kModelSchema == "jacek_native_model/v1",
          "The model artifact schema changed.");
  require(model::kFeatureSchema ==
              "canonical-edges316-onehot-true-turn-distance105x8-v1",
          "The model feature schema changed.");
  require(model::kModelSha256 ==
              "b00b9d543fbc7d58fe342d5340cbdeb4e3e2d6d522938ef2b8e0aaea18193d14",
          "The retained native JSON model changed without review.");
  require(model::kSchemaSha256 ==
              "dd36c1b2800620fab1d5dc88afe95fcbb13864d581a18f01d26b3e1c3a4a6dfd",
          "The frozen model schema changed without review.");
  require(model::kPackedSha256 ==
              "e2304195d491d7b2d5ae1334a8341b38d67d315073accc37915885ede3c6a2cb",
          "The deployed packed weights changed without review.");
  require(model::kPackedByteCount * 8U >= model::kWeightCount * 3U,
          "The packed model is too short for its declared weights.");
  require(candidate::default_model().model_sha256() == model::kModelSha256 &&
              candidate::default_model().packed_sha256() ==
                  model::kPackedSha256,
          "Runtime model validation must reproduce both embedded hashes.");

  std::string tampered(model::kPackedWeights);
  require(!tampered.empty(), "The retained native checkpoint cannot be empty.");
  tampered.front() = tampered.front() == 'A' ? 'B' : 'A';
  require_invalid_argument(
      [&] {
        (void)candidate::QuantizedModel(
            tampered, model::kScale1, model::kScale2, model::kScale3,
            model::kModelSchema, model::kFeatureSchema, model::kModelSha256,
            model::kModelSha256, false, model::kPackedSha256);
      },
      "A one-byte packed-weight mutation must fail the retained SHA-256.");
}

void partial_quantized_and_float_evaluation_agree() {
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  auto topology = std::make_shared<candidate::SearchTopology>(state.config);
  const candidate::SearchPosition root(topology, state);
  candidate::SearchPosition child = root;
  std::uint8_t slot = 0;
  require(child.slot_for_move(ps::Move{{3, 5}}, slot),
          "The partial-evaluation fixture move must be legal.");
  child.make_move(slot);
  require(child.to_move() == ps::Player::Two,
          "The partial-evaluation fixture must hand off the turn.");

  candidate::NeuralEvaluator evaluator(topology, candidate::default_model());
  const candidate::PreparedEvaluation prepared =
      evaluator.prepare(root, child.to_move());
  const float partial = evaluator.evaluate_child(prepared, child);
  const float full = evaluator.evaluate(child);
  require(std::abs(partial - full) <= 1e-6F,
          "Partial first-layer reevaluation must equal full reevaluation.");

  const candidate::QuantizedModel &quantized = candidate::default_model();
  const std::span<const std::int8_t> weights = quantized.weights();
  std::vector<float> first(candidate::kFeatureCount * candidate::kHiddenOne);
  std::vector<float> second(candidate::kHiddenOne * candidate::kHiddenTwo);
  std::vector<float> output(candidate::kHiddenTwo);
  std::size_t cursor = 0;
  for (float &value : first) {
    value = static_cast<float>(weights[cursor++]) * quantized.scale_one();
  }
  for (float &value : second) {
    value = static_cast<float>(weights[cursor++]) * quantized.scale_two();
  }
  for (float &value : output) {
    value = static_cast<float>(weights[cursor++]) * quantized.scale_three();
  }
  require(cursor == weights.size(),
          "The float parity fixture must consume every packed weight.");
  const candidate::DenseFeatures features = candidate::encode_features(state);
  const float unpacked = candidate::evaluate_float_model(
      features, candidate::FloatModelView{first, second, output});
  require(std::abs(unpacked - quantized.evaluate(features)) <= 1e-6F,
          "Packed quantized inference must equal unpacked float inference.");
  constexpr float kIndependentPythonInitialValue = -0.0161130223423243F;
  require(std::abs(quantized.evaluate(features) -
                   kIndependentPythonInitialValue) <= 2e-6F,
          "C++ inference must match the retained JSON's independent Python "
          "initial-state golden value.");
}

candidate::GenerationResult generate(
    const ps::GameState &state,
    candidate::GeneratorConfig config = candidate::GeneratorConfig{}) {
  auto topology = std::make_shared<candidate::SearchTopology>(state.config);
  const candidate::SearchPosition root(topology, state);
  candidate::CompleteTurnGenerator generator(topology, root, config);
  return generator.run();
}

std::vector<std::string> action_strings(
    const candidate::GenerationResult &generated) {
  std::vector<std::string> result;
  result.reserve(generated.actions.size());
  for (const candidate::CompleteTurnAction &action : generated.actions) {
    result.push_back(action.encoded);
  }
  return result;
}

void complete_turn_generator_enforces_boundary_and_schedule_contracts() {
  candidate::GeneratorConfig forced_config;
  forced_config.shuffle_seed = 17;
  const candidate::GenerationResult forced =
      generate(forced_two_move_rebound(), forced_config);
  require(forced.actions.size() == 1 && forced.actions.front().encoded == "00",
          "A forced rebound must produce its exact complete turn.");

  candidate::GeneratorConfig config;
  config.max_actions = candidate::kMaximumActions;
  config.max_partial_paths = 20;
  config.shuffle_seed = 0x91d4e706d7b84c2bULL;
  const ps::GameState dense = dense_rebound_state();
  const candidate::GenerationResult first = generate(dense, config);
  const candidate::GenerationResult second = generate(dense, config);
  require(!first.actions.empty() &&
              first.actions.size() <= candidate::kMaximumActions,
          "The generator must return between one and 250 actions.");
  require(action_strings(first) == action_strings(second),
          "A fixed shuffle seed must reproduce action ordering.");
  require(first.stats.explored_partial_paths == 20 &&
              first.stats.lifo_extractions == 18 &&
              first.stats.fifo_extractions == 2,
          "Twenty partial paths must exercise the required 9:1 deque schedule.");
  require(first.stats.lifo_extractions + first.stats.fifo_extractions ==
              first.stats.explored_partial_paths,
          "Every explored partial path must have one recorded extraction.");
  require(first.stats.truncations == 1 && !first.exhaustive,
          "The synthetic partial-path cap must be observable.");

  auto topology = std::make_shared<candidate::SearchTopology>(dense.config);
  std::vector<candidate::PositionKey> boundaries;
  for (const candidate::CompleteTurnAction &action : first.actions) {
    require(!action.encoded.empty(), "A generated complete turn cannot be empty.");
    ps::GameState replayed = dense;
    candidate::apply_encoded_turn(replayed, action.encoded);
    const candidate::SearchPosition replay_position(topology, replayed);
    require(replay_position.position_key() == action.result.position_key(),
            "A generated result must match replaying its encoded turn.");
    require(std::find(boundaries.begin(), boundaries.end(),
                      action.result.position_key()) == boundaries.end(),
            "Two retained paths must not share a full boundary PositionKey.");
    boundaries.push_back(action.result.position_key());
  }
  require(first.stats.completed_actions ==
              first.actions.size() + first.stats.duplicate_boundaries,
          "Boundary duplicates must be counted instead of retained.");

  candidate::GeneratorConfig full_cap;
  full_cap.max_actions = candidate::kMaximumActions;
  full_cap.max_partial_paths = candidate::kMaximumPartialPaths;
  full_cap.shuffle_seed = 0x147b82f3956c20adULL;
  const candidate::GenerationResult capped = generate(dense, full_cap);
  require(capped.actions.size() == candidate::kMaximumActions &&
              capped.stats.truncations == 1,
          "The dense fixture must exercise the exact 250-action hard cap.");
}

candidate::CompleteTurnAction cap_one_action(const ps::GameState &state,
                                              std::uint64_t seed) {
  candidate::GeneratorConfig config;
  config.max_actions = 1;
  config.max_partial_paths = 5'000;
  config.shuffle_seed = seed;
  candidate::GenerationResult generated = generate(state, config);
  require(generated.actions.size() == 1,
          "A cap-one generator must retain exactly one complete turn.");
  return std::move(generated.actions.front());
}

void tactical_top_k_is_global_and_seed_independent() {
  ps::GameState immediate = make_clean_state_at({4, 1}, ps::Player::One);
  block_edges_except(immediate, immediate.ball, {{4, 0}, {4, 2}});

  ps::GameState forced = make_clean_state_at({4, 6}, ps::Player::One);
  forced.visit_count[{4, 4}] = 1;
  block_edges_except(forced, {4, 6}, {{4, 5}, {5, 6}});
  block_edges_except(forced, {4, 5}, {{4, 6}, {4, 4}});
  block_edges_except(forced, {4, 4}, {{4, 5}});

  ps::GameState safe = make_clean_state_at({4, 10}, ps::Player::One);
  block_edges_except(safe, safe.ball, {{4, 11}, {3, 9}});

  ps::GameState safe_over_own =
      make_clean_state_at({4, 11}, ps::Player::One);
  block_edges_except(safe_over_own, safe_over_own.ball, {{4, 12}, {4, 10}});
  block_edges_except(safe_over_own, {4, 10},
                     {{4, 11}, {3, 9}, {4, 9}, {5, 9}});

  ps::GameState danger_over_own =
      make_clean_state_at({4, 11}, ps::Player::One);
  block_edges_except(danger_over_own, danger_over_own.ball,
                     {{4, 12}, {4, 10}});

  for (const std::uint64_t seed : {1ULL, 17ULL, 0x9e3779b97f4a7c15ULL}) {
    const candidate::CompleteTurnAction goal = cap_one_action(immediate, seed);
    require(goal.tactical == candidate::TacticalClass::ImmediateGoal &&
                goal.encoded == "0",
            "ImmediateGoal must outrank every non-winning cap-one action.");

    const candidate::CompleteTurnAction cutoff = cap_one_action(forced, seed);
    require(cutoff.tactical == candidate::TacticalClass::ForcedCutoff &&
                cutoff.encoded == "0",
            "ForcedCutoff must outrank an ordinary safe handoff.");

    const candidate::CompleteTurnAction handoff = cap_one_action(safe, seed);
    require(handoff.tactical == candidate::TacticalClass::SafeHandoff &&
                handoff.encoded == "7",
            "SafeHandoff must outrank an opponent's immediate goal.");

    const candidate::CompleteTurnAction own_safe =
        cap_one_action(safe_over_own, seed);
    require(own_safe.tactical == candidate::TacticalClass::SafeHandoff &&
                own_safe.encoded == "0",
            "SafeHandoff must outrank an available own goal.");

    const candidate::CompleteTurnAction dangerous =
        cap_one_action(danger_over_own, seed);
    require(dangerous.tactical ==
                    candidate::TacticalClass::OpponentImmediateGoal &&
                dangerous.encoded == "0",
            "OpponentImmediateGoal must still outrank OwnGoal at cap one.");
  }
}

void tactical_witnesses_survive_ordinary_path_truncation() {
  ps::GameState forced = make_clean_state_at({4, 6}, ps::Player::One);
  forced.visit_count[{4, 5}] = 1;
  forced.visit_count[{4, 4}] = 1;
  forced.visit_count[{6, 4}] = 1;
  block_edges_except(forced, {4, 6}, {{4, 5}, {5, 6}});
  block_edges_except(forced, {4, 5}, {{4, 6}, {4, 4}});
  block_edges_except(forced, {4, 4}, {{4, 5}, {5, 4}});
  block_edges_except(forced, {5, 4}, {{4, 4}, {6, 4}});
  block_edges_except(forced, {6, 4}, {{5, 4}});
  candidate::GeneratorConfig config;
  config.max_actions = 1;
  config.max_partial_paths = 1;
  config.shuffle_seed = 17;
  const candidate::GenerationResult generated = generate(forced, config);
  require(generated.actions.size() == 1 &&
              generated.actions.front().encoded == "002" &&
              generated.actions.front().tactical ==
                  candidate::TacticalClass::ForcedCutoff,
          "The bounded tactical pass must preserve a forced cutoff when the "
          "ordinary sampler truncates after its first extraction.");
  require(generated.stats.proof_paths > 1,
          "The forced-cutoff witness must require a multi-path proof pass.");
  require(generated.stats.proof_classes >= 1,
          "The tactical proof pass must count the retained witness class.");
  require(generated.stats.explored_partial_paths == 1,
          "Tactical proof work must be separate from ordinary sampler work.");

  candidate::GeneratorConfig dense_config = config;
  dense_config.max_actions = 8;
  const candidate::GenerationResult dense =
      generate(dense_rebound_state(), dense_config);
  require(dense.stats.proof_paths == candidate::kProofPaths &&
              dense.stats.proof_truncated,
          "A bounded tactical witness pass must expose its own truncation.");
}

void closed_branches_backtrack_and_reflection_preserves_generation() {
  ps::GameState closed = make_clean_state_at({4, 6});
  closed.visit_count[{4, 5}] = 1;
  closed.visit_count[{4, 4}] = 1;
  block_edges_except(closed, {4, 6}, {{4, 5}});
  block_edges_except(closed, {4, 5}, {{4, 6}, {4, 4}, {5, 5}});
  block_edges_except(closed, {4, 4}, {{4, 5}});
  const ps::GameState snapshot = closed;
  candidate::GeneratorConfig closed_config;
  closed_config.max_actions = 8;
  closed_config.max_partial_paths = 2;
  closed_config.shuffle_seed = 91;
  const candidate::GenerationResult branches = generate(closed, closed_config);
  const std::vector<std::string> branch_actions = action_strings(branches);
  require(std::find(branch_actions.begin(), branch_actions.end(), "00") !=
                  branch_actions.end() &&
              std::find(branch_actions.begin(), branch_actions.end(), "02") !=
                  branch_actions.end(),
          "A closed rebound branch must not erase its safe sibling.");
  require(branches.stats.explored_partial_paths == 2 && branches.exhaustive,
          "The closed-branch fixture must finish in exactly two extractions.");
  require(same_state(closed, snapshot),
          "Generation must not mutate its source state while backtracking.");

  ps::GameState asymmetric = make_clean_state_at({3, 6});
  block_edges_except(asymmetric, asymmetric.ball,
                     {{3, 5}, {2, 5}, {2, 6}});
  const ps::GameState twin = reflected_state(asymmetric);
  candidate::GeneratorConfig mirror_config;
  mirror_config.max_actions = 8;
  mirror_config.max_partial_paths = 32;
  mirror_config.shuffle_seed = 7;
  std::vector<std::string> left =
      action_strings(generate(asymmetric, mirror_config));
  std::vector<std::string> right = action_strings(generate(twin, mirror_config));
  for (std::string &action : left) {
    action = reflected_action(std::move(action));
  }
  std::sort(left.begin(), left.end());
  std::sort(right.begin(), right.end());
  require(left == right,
          "Horizontal reflection must preserve the complete-turn action set.");
}

candidate::SearchConfig one_expansion_config() {
  candidate::SearchConfig config;
  config.max_actions = 16;
  config.max_partial_paths = 1'000;
  config.max_tree_nodes = 64;
  config.max_expansions = 1;
  config.shuffle_seed = 29;
  return config;
}

void minimax_values_are_mover_relative_and_distance_sensitive() {
  const float deep_child_loss = candidate::terminal_value(
      ps::Player::One, ps::Player::Two, 200);
  require(deep_child_loss == -8'000.0F &&
              candidate::proven_action_win(true, -deep_child_loss),
          "A solved win beyond 100 complete turns must still propagate.");

  ps::GameState win = make_clean_state_at({4, 1}, ps::Player::One);
  block_edges_except(win, win.ball, {{4, 0}});
  const candidate::SearchResult winning =
      candidate::choose_complete_turn(win, one_expansion_config());
  require(winning.encoded == "0" &&
              winning.value == candidate::kMateScore - 10.0F,
          "An immediate attacking goal must be a positive ply-one mate.");

  ps::GameState loss = make_clean_state_at({4, 11}, ps::Player::One);
  block_edges_except(loss, loss.ball, {{4, 12}});
  const candidate::SearchResult losing =
      candidate::choose_complete_turn(loss, one_expansion_config());
  require(losing.encoded == "4" &&
              losing.value == -candidate::kMateScore + 10.0F,
          "An immediate own goal must be a negative ply-one mate.");
}

const candidate::QuantizedModel &zero_model() {
  static const candidate::QuantizedModel result(
      "", 1.0F, 1.0F, 1.0F, model::kModelSchema,
      model::kFeatureSchema, model::kModelSha256, model::kModelSha256, true);
  return result;
}

const candidate::RootActionStat &root_action(
    const candidate::SearchResult &result, std::string_view encoded) {
  const auto found = std::find_if(
      result.root_actions.begin(), result.root_actions.end(),
      [&](const candidate::RootActionStat &action) {
        return action.encoded == encoded;
      });
  require(found != result.root_actions.end(),
          "The expected root action is missing from the fixture.");
  return *found;
}

void search_uses_fpu_uct_visits_and_lexical_ties() {
  candidate::SearchConfig config;
  config.max_actions = 8;
  config.max_partial_paths = 64;
  config.max_tree_nodes = 256;
  config.max_expansions = 1;
  config.shuffle_seed = 0;
  config.exploration_constant = candidate::kExplorationConstant;
  config.first_play_urgency = candidate::kFirstPlayUrgency;
  config.model = &zero_model();
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const candidate::SearchResult tied =
      candidate::choose_complete_turn(state, config);
  require(tied.encoded == "0" && tied.root_actions.size() == 8,
          "Equal values and visits must choose the lexical action key.");
  require(std::all_of(
              tied.root_actions.begin(), tied.root_actions.end(),
              [](const candidate::RootActionStat &action) {
                return action.value == 0.0F && action.initial_value == 0.0F &&
                       action.visits == 1 && action.selection_visits == 0;
              }),
          "The tie fixture must start every root action with one final visit.");

  config.max_expansions = 4;
  const candidate::SearchResult allocated =
      candidate::choose_complete_turn(state, config);
  const candidate::RootActionStat &zero = root_action(allocated, "0");
  const candidate::RootActionStat &one = root_action(allocated, "1");
  const candidate::RootActionStat &two = root_action(allocated, "2");
  require(zero.selection_visits == 2 && one.selection_visits == 1 &&
              two.selection_visits == 0,
          "FPU must visit 0 then 1 before C=0.80 revisits lexical action 0.");
  require(zero.visits == 3 && one.visits == 2 && two.visits == 1 &&
              allocated.encoded == "0",
          "Unequal visits must feed final value+log(visits) selection.");
}

void fixed_work_search_is_deterministic_legal_and_capped() {
  candidate::SearchConfig config;
  config.max_actions = 16;
  config.max_partial_paths = 1'000;
  config.max_tree_nodes = 256;
  config.max_expansions = 8;
  config.shuffle_seed = 0x4e8b215e3bca93d7ULL;
  config.exploration_constant = candidate::kExplorationConstant;
  config.first_play_urgency = candidate::kFirstPlayUrgency;
  const ps::GameState state = ps::make_initial_state(codingame_rules());
  const candidate::SearchResult first =
      candidate::choose_complete_turn(state, config);
  const candidate::SearchResult second =
      candidate::choose_complete_turn(state, config);
  require(!first.encoded.empty() && first.encoded == second.encoded,
          "Fixed-work search must choose deterministically.");
  require(first.encoded == "7" && first.stats.expansions == 8 &&
              first.stats.generated_children == 80 &&
              first.stats.child_evaluations == 80 &&
              first.stats.completed_actions == 90 &&
              first.stats.duplicate_boundaries == 8 &&
              first.stats.tactical_actions == 0 &&
              first.stats.generator_partial_paths == 14 &&
              first.stats.tree_nodes == 81,
          "Duplicate filtering must preserve the frozen fixed-work action and "
          "all pre-optimization work counters.");
  require(first.stats.root_generator_truncations == 0 &&
              first.stats.nonroot_generator_truncations == 1 &&
              first.stats.generator_truncations == 1 &&
              first.stats.max_complete_turn_depth == 5 &&
              !first.stats.tree_cap_reached &&
              first.stats.expansion_cap_reached,
          "Fixed-work telemetry must distinguish depth and cap causes.");
  require(first.root_actions.size() == second.root_actions.size(),
          "Repeated fixed-work searches must retain the same root actions.");
  for (std::size_t index = 0; index < first.root_actions.size(); ++index) {
    const auto &left = first.root_actions[index];
    const auto &right = second.root_actions[index];
    require(left.encoded == right.encoded && left.value == right.value &&
                left.initial_value == right.initial_value &&
                left.visits == right.visits &&
                left.selection_visits == right.selection_visits &&
                left.tactical == right.tactical,
            "Repeated fixed-work root telemetry must be bit-exact.");
  }
  require(first.root_actions.size() <= config.max_actions &&
              !first.root_actions.empty(),
          "Root generation must obey the configured action cap.");
  ps::GameState replayed = state;
  candidate::apply_encoded_turn(replayed, first.encoded);
  require(ps::is_terminal(replayed) || replayed.to_move != state.to_move,
          "Fixed-work search must return a rebound-complete action.");
  require(first.stats.expansions <= config.max_expansions &&
              first.stats.tree_nodes <= config.max_tree_nodes &&
              first.stats.child_evaluations <=
                  first.stats.expansions * config.max_actions,
          "Search diagnostics must respect every hard work cap.");
  require(first.stats.generator_fifo_extractions +
                  first.stats.generator_lifo_extractions ==
              first.stats.generator_partial_paths,
          "Search must account for every generator extraction.");

  candidate::SearchConfig forced_config = config;
  forced_config.max_expansions = 1;
  const candidate::SearchResult forced =
      candidate::choose_complete_turn(forced_two_move_rebound(), forced_config);
  require(forced.encoded == "00",
          "Search must preserve an exact forced two-edge rebound.");

  candidate::SearchConfig truncated_config = config;
  truncated_config.max_actions = 8;
  truncated_config.max_partial_paths = 1;
  truncated_config.max_tree_nodes = 64;
  truncated_config.max_expansions = 2;
  const candidate::SearchResult truncated = candidate::choose_complete_turn(
      dense_rebound_state(), truncated_config);
  require(truncated.stats.root_generator_truncations == 1 &&
              truncated.stats.nonroot_generator_truncations == 1 &&
              truncated.stats.generator_truncations == 2 &&
              truncated.stats.max_complete_turn_depth == 2,
          "Search telemetry must separate root and non-root generator "
          "truncations while retaining their aggregate.");
}

void timed_search_returns_a_complete_replayable_action() {
  const ps::GameState original = ps::make_initial_state(codingame_rules());
  ps::GameState chosen = original;
  const std::string action = candidate::choose_complete_turn(chosen, 1);
  require(!action.empty(), "Timed search must return a legal fallback.");
  ps::GameState replayed = original;
  candidate::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed),
          "The timed action must reconstruct the chosen state exactly.");
}

}  // namespace

int main() {
  try {
    constants_freeze_the_direct_contract();
    uct_fpu_and_final_visit_formulas_are_literal();
    directions_round_trip_exactly();
    own_goals_and_blocked_mover_follow_contest_rules();
    encoded_turn_validation_is_atomic();
    initial_features_match_the_article_contract();
    feature_order_and_player_rotation_are_bit_exact();
    model_artifact_identity_is_frozen();
    partial_quantized_and_float_evaluation_agree();
    complete_turn_generator_enforces_boundary_and_schedule_contracts();
    tactical_top_k_is_global_and_seed_independent();
    tactical_witnesses_survive_ordinary_path_truncation();
    closed_branches_backtrack_and_reflection_preserves_generation();
    minimax_values_are_mover_relative_and_distance_sensitive();
    search_uses_fpu_uct_visits_and_lexical_ties();
    fixed_work_search_is_deterministic_legal_and_capped();
    timed_search_returns_a_complete_replayable_action();
    std::cout << "jacek_native_bfm submission tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek_native_bfm submission test failure: "
              << error.what() << '\n';
    return 1;
  }
}
