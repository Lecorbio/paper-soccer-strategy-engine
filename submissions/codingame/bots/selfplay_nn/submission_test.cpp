#define PAPER_SOCCER_SELFPLAY_NN_NO_MAIN
#include "submission.cpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::selfplay_nn;

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

std::vector<std::string_view> split_transcript(std::string_view transcript) {
  std::vector<std::string_view> turns;
  std::size_t start = 0;
  while (start <= transcript.size()) {
    const std::size_t separator = transcript.find('/', start);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    require(end != start, "Transcript contains an empty turn.");
    turns.push_back(transcript.substr(start, end - start));
    if (separator == std::string_view::npos) {
      break;
    }
    start = separator + 1;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  for (const std::string_view turn : split_transcript(transcript)) {
    cg::apply_encoded_turn(state, turn);
  }
  return state;
}

void directions_round_trip_exactly() {
  const ps::Point origin{4, 6};
  constexpr std::array<ps::Point, 8> expected{{
      {4, 5}, {5, 5}, {5, 6}, {5, 7},
      {4, 7}, {3, 7}, {3, 6}, {3, 5},
  }};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const char direction = static_cast<char>('0' + index);
    require(cg::decode_direction(origin, direction).to == expected[index],
            "Direction decoding should match the CodinGame compass table.");
    require(cg::encode_direction(origin, expected[index]) == direction,
            "Direction encoding should invert every valid direction.");
  }
  require_invalid_argument(
      [&] { (void)cg::decode_direction(origin, '8'); },
      "Direction eight must never be accepted or emitted.");
}

void own_goals_and_blocked_mover_follow_contest_rules() {
  ps::GameState north = make_clean_state_at({4, 1}, ps::Player::Two);
  north = ps::apply_move(north, {{4, 0}});
  require(ps::winner(north) == ps::Player::One,
          "Player Two's north own goal should award Player One.");

  ps::GameState south = make_clean_state_at({4, 11}, ps::Player::One);
  south = ps::apply_move(south, {{4, 12}});
  require(ps::winner(south) == ps::Player::Two,
          "Player One's south own goal should award Player Two.");

  ps::GameState blocked = make_clean_state_at({4, 6}, ps::Player::One);
  block_edges_except(blocked, {4, 5}, {{4, 6}});
  blocked = ps::apply_move(blocked, {{4, 5}});
  require(ps::winner(blocked) == ps::Player::Two,
          "A blocked landing should immediately defeat the mover.");
}

void encoded_turn_validation_is_atomic() {
  ps::GameState rebound = forced_two_move_rebound();
  const ps::GameState snapshot = rebound;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(rebound, "0"); },
      "A turn may not stop during a mandatory rebound.");
  require(same_state(rebound, snapshot),
          "A rejected early turn should not partially mutate state.");

  cg::apply_encoded_turn(rebound, "00");
  require(rebound.ball == ps::Point{4, 2} &&
              rebound.to_move == ps::Player::Two,
          "A complete rebound sequence should hand possession over.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::GameState initial_snapshot = initial;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(initial, "00"); },
      "A turn may not include an opponent's next move.");
  require(same_state(initial, initial_snapshot),
          "A rejected extra move should not partially mutate state.");
}

void timed_search_returns_a_complete_replayable_action() {
  const ps::GameState original = ps::make_initial_state(codingame_rules());
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(!action.empty(), "Timed search should return an action.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed),
          "The timed action should reconstruct the chosen state.");
}

void normal_fallback_finishes_mandatory_rebounds() {
  const ps::GameState original = forced_two_move_rebound();
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(action == "00",
          "The normal V2 path should finish the forced rebound action.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed) &&
              replayed.to_move == ps::Player::Two,
          "The fallback action should be legal and rebound-complete.");
}

void exact_opening_and_replay_corrections_are_safe() {
  ps::GameState guarded = reconstruct("1");
  const ps::GameState guarded_before = guarded;
  std::string action;
  require(cg::try_replay_correction(guarded, 1, "1", action) &&
              action == "1",
          "The exact Player-1 opening hole should select direction one.");
  ps::GameState guarded_replayed = guarded_before;
  cg::apply_encoded_turn(guarded_replayed, action);
  require(same_state(guarded, guarded_replayed),
          "The guarded opening action should reconstruct exactly.");

  ps::GameState deltaspace = reconstruct("0/6");
  const ps::GameState deltaspace_before = deltaspace;
  require(cg::try_replay_correction(deltaspace, 0, "0/6", action) &&
              action == "5",
          "The exact Player-0 Deltaspace root should select direction five.");
  ps::GameState deltaspace_replayed = deltaspace_before;
  cg::apply_encoded_turn(deltaspace_replayed, action);
  require(same_state(deltaspace, deltaspace_replayed),
          "The guarded Deltaspace action should reconstruct exactly.");

  ps::GameState replay = reconstruct("0/2/7/45");
  const ps::GameState replay_before = replay;
  require(cg::try_replay_correction(replay, 0, "0/2/7/45", action) &&
              action == "7",
          "The verified Snekkers replay should select direction seven.");
  ps::GameState replayed = replay_before;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(replay, replayed),
          "The exact replay correction should reconstruct exactly.");
}

void interrupted_search_preserves_a_complete_action() {
  const ps::GameState state = forced_two_move_rebound();
  cg::SearchConfig config;
  config.max_nodes = 1;
  config.max_time_ms = 0;
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  require(search.stats().budget_exhausted && !moves.empty(),
          "A one-node search should return its complete fallback action.");

  ps::GameState replayed = state;
  std::string encoded;
  for (const ps::Move move : moves) {
    encoded.push_back(cg::encode_direction(replayed.ball, move.to));
    replayed = ps::apply_move(replayed, move);
  }
  ps::GameState verified = state;
  cg::apply_encoded_turn(verified, encoded);
  require(same_state(replayed, verified),
          "The interrupted search fallback should replay exactly.");
}

void model_dimensions_and_search_hooks_are_live() {
  require(cg::neural_model::kInputCount == 423 &&
              cg::neural_model::kEdgeCount == 316 &&
              cg::neural_model::kVertexCount == 105 &&
              cg::neural_model::kHiddenOne == 16 &&
              cg::neural_model::kHiddenTwo == 16 &&
              cg::neural_model::kPolicyCount == 8,
          "The generated policy/value model dimensions changed.");

  const ps::GameState state = ps::make_initial_state(codingame_rules());
  cg::SearchConfig config;
  config.max_turn_depth = 1;
  config.max_nodes = 20'000;
  config.neural_value_blend_percent = 15;
  config.neural_policy_order_weight = 1'500;
  cg::CompleteTurnSearch search(state, config);
  const auto [value, policy] = search.neural_evaluation();
  require(std::isfinite(value), "The neural value output must be finite.");
  require(std::all_of(policy.begin(), policy.end(),
                      [](float logit) { return std::isfinite(logit); }),
          "Every neural policy output must be finite.");
  (void)search.run();
  require(search.stats().neural_evaluations > 1 &&
              search.stats().neural_policy_orderings > 0,
          "Both neural heads must participate in search.");
}

ps::Point rotate_point(ps::Point point) {
  return {8 - point.x, 12 - point.y};
}

ps::GameState rotate_and_swap(const ps::GameState &state) {
  ps::GameState result = state;
  result.ball = rotate_point(state.ball);
  result.to_move = ps::opponent(state.to_move);
  result.path.clear();
  for (const ps::Point point : state.path) {
    result.path.push_back(rotate_point(point));
  }
  result.used_segments.clear();
  for (const ps::Segment &edge : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{rotate_point(edge.a), rotate_point(edge.b)});
  }
  result.visit_count.clear();
  for (const auto &[point, count] : state.visit_count) {
    result.visit_count[rotate_point(point)] = count;
  }
  return result;
}

void neural_features_are_player_rotation_invariant() {
  const ps::GameState state = reconstruct("2/5/4/1/3/4/3/57");
  const ps::GameState rotated = rotate_and_swap(state);
  cg::SearchConfig config;
  config.neural_value_blend_percent = 1;
  cg::CompleteTurnSearch original_search(state, config);
  cg::CompleteTurnSearch rotated_search(rotated, config);
  const auto [original_value, original_policy] =
      original_search.neural_evaluation();
  const auto [rotated_value, rotated_policy] =
      rotated_search.neural_evaluation();
  require(std::abs(original_value - rotated_value) < 1e-5F,
          "Canonical player rotation changed the neural value.");
  for (std::size_t index = 0; index < original_policy.size(); ++index) {
    require(std::abs(original_policy[index] - rotated_policy[index]) < 1e-5F,
            "Canonical player rotation changed a policy logit.");
  }
}

void require_golden_logits(
    const ps::GameState &state, float expected_value,
    const std::array<float, 8> &expected_policy) {
  cg::SearchConfig config;
  config.neural_value_blend_percent = 1;
  cg::CompleteTurnSearch search(state, config);
  const auto [actual_value, actual_policy] = search.neural_evaluation();
  require(std::abs(actual_value - expected_value) < 2e-5F,
          "Python and C++ produced different quantized value logits.");
  for (std::size_t index = 0; index < actual_policy.size(); ++index) {
    require(std::abs(actual_policy[index] - expected_policy[index]) < 2e-5F,
            "Python and C++ produced different quantized policy logits.");
  }
}

void python_and_cpp_golden_logits_match() {
  require_golden_logits(
      ps::make_initial_state(codingame_rules()), 0.256164670F,
      {{0.240548730F, 0.366741896F, -0.00642091036F, -0.0649466291F,
        -0.154909372F, 0.00837692618F, 0.0600975901F, 0.289233059F}});
  require_golden_logits(
      reconstruct("2"), 0.204200655F,
      {{0.250912279F, 0.385505646F, -0.00363897160F, -0.0556779355F,
        -0.115268059F, 0.00380627811F, 0.0759369433F, 0.286905974F}});
  require_golden_logits(
      reconstruct("2/5/4/1/3/4/3/57"), -0.149147779F,
      {{0.251313686F, 0.463242471F, 0.0852589458F, -0.0189857334F,
        -0.0391605422F, 0.0262221247F, 0.109565511F, 0.349172145F}});
}

}  // namespace

int main() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"directions_round_trip_exactly", directions_round_trip_exactly},
      {"own_goals_and_blocked_mover_follow_contest_rules",
       own_goals_and_blocked_mover_follow_contest_rules},
      {"encoded_turn_validation_is_atomic",
       encoded_turn_validation_is_atomic},
      {"timed_search_returns_a_complete_replayable_action",
       timed_search_returns_a_complete_replayable_action},
      {"normal_fallback_finishes_mandatory_rebounds",
       normal_fallback_finishes_mandatory_rebounds},
      {"exact_opening_and_replay_corrections_are_safe",
       exact_opening_and_replay_corrections_are_safe},
      {"interrupted_search_preserves_a_complete_action",
       interrupted_search_preserves_a_complete_action},
      {"model_dimensions_and_search_hooks_are_live",
       model_dimensions_and_search_hooks_are_live},
      {"neural_features_are_player_rotation_invariant",
       neural_features_are_player_rotation_invariant},
      {"python_and_cpp_golden_logits_match",
       python_and_cpp_golden_logits_match},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << '\n' << (tests.size() - static_cast<std::size_t>(failures))
            << '/' << tests.size() << " submission tests passed.\n";
  return failures == 0 ? 0 : 1;
}
