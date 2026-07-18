#define PAPER_SOCCER_CODINGAME_NO_MAIN
#include "paper_soccer_alpha_beta.cpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::codingame;

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

bool same_state(const ps::GameState &lhs, const ps::GameState &rhs) {
  return lhs.config.width == rhs.config.width &&
         lhs.config.height == rhs.config.height &&
         lhs.config.goal_rule == rhs.config.goal_rule &&
         lhs.config.blocked_rule == rhs.config.blocked_rule &&
         lhs.ball == rhs.ball && lhs.to_move == rhs.to_move &&
         lhs.status == rhs.status && lhs.path == rhs.path &&
         lhs.used_segments == rhs.used_segments &&
         lhs.visit_count == rhs.visit_count;
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

void own_goals_are_legal_and_awarded_to_the_other_player() {
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
  const std::string action =
      cg::complete_turn_from_plan(blocked, {ps::Move{{4, 5}}});
  require(action == "0" && ps::winner(blocked) == ps::Player::Two,
          "A blocked landing should immediately defeat the mover.");
}

void turn_validation_rejects_early_and_extra_moves() {
  ps::GameState rebound = forced_two_move_rebound();
  const ps::GameState snapshot = rebound;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(rebound, "0"); },
      "A turn may not stop during a mandatory rebound.");
  require(same_state(rebound, snapshot),
          "A rejected turn should not partially mutate the game.");

  cg::apply_encoded_turn(rebound, "00");
  require(rebound.ball == ps::Point{4, 2} &&
              rebound.to_move == ps::Player::Two,
          "A complete rebound sequence should hand possession over.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(initial, "00"); },
      "A turn may not include an opponent's next move.");
}

void composer_finishes_rebounds_and_stops_at_the_turn_boundary() {
  ps::GameState rebound = forced_two_move_rebound();
  const std::string forced =
      cg::complete_turn_from_plan(rebound, {ps::Move{{4, 3}}});
  require(forced == "00" && rebound.to_move == ps::Player::Two,
          "The fallback should finish a rebound when the plan runs out.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const std::string one_turn = cg::complete_turn_from_plan(
      initial, {ps::Move{{4, 5}}, ps::Move{{4, 4}}});
  require(one_turn == "0" && initial.to_move == ps::Player::Two,
          "The composer must not emit the opponent portion of a deeper plan.");

  ps::GameState goal = make_clean_state_at({4, 1}, ps::Player::One);
  const std::string terminal = cg::complete_turn_from_plan(
      goal, {ps::Move{{4, 0}}, ps::Move{{4, 1}}});
  require(terminal == "0" && ps::winner(goal) == ps::Player::One,
          "The composer should stop immediately after a goal.");
}

void timed_search_produces_a_complete_replayable_turn() {
  const ps::GameState original = ps::make_initial_state(codingame_rules());
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(action.size() == 1,
          "Every initial move should end the first player's turn.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed),
          "The emitted action should reconstruct the chosen game state.");
}

void deterministic_turns_remain_replayable() {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  int turns = 0;
  while (!ps::is_terminal(state) && turns < 24) {
    const ps::GameState before = state;
    const std::string action = cg::complete_turn_from_plan(state, {});
    ps::GameState replayed = before;
    cg::apply_encoded_turn(replayed, action);
    require(same_state(state, replayed),
            "Every composed turn should pass the protocol verifier mirror.");
    ++turns;
  }
  require(turns > 0, "The replayability check should execute at least one turn.");
}

}  // namespace

int main() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"directions_round_trip_exactly", directions_round_trip_exactly},
      {"own_goals_are_legal_and_awarded_to_the_other_player",
       own_goals_are_legal_and_awarded_to_the_other_player},
      {"turn_validation_rejects_early_and_extra_moves",
       turn_validation_rejects_early_and_extra_moves},
      {"composer_finishes_rebounds_and_stops_at_the_turn_boundary",
       composer_finishes_rebounds_and_stops_at_the_turn_boundary},
      {"timed_search_produces_a_complete_replayable_turn",
       timed_search_produces_a_complete_replayable_turn},
      {"deterministic_turns_remain_replayable",
       deterministic_turns_remain_replayable},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << "\n";
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << "\n";
    }
  }
  std::cout << "\n" << (tests.size() - static_cast<std::size_t>(failures))
            << "/" << tests.size() << " submission tests passed.\n";
  return failures == 0 ? 0 : 1;
}
