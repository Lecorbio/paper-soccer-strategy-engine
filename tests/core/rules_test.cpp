#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "papersoccer/debug.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool contains_move(const std::vector<ps::Move> &moves, ps::Point point) {
  for (const ps::Move &move : moves) {
    if (move.to == point) {
      return true;
    }
  }
  return false;
}

std::vector<ps::Point> goal_destinations(const ps::GameState &state) {
  std::vector<ps::Point> result;
  for (const ps::Move move : ps::legal_moves(state)) {
    if (ps::is_goal_point(state.config, move.to)) {
      result.push_back(move.to);
    }
  }
  return result;
}

ps::GameState make_clean_state_at(
    ps::Point point, ps::Player to_move = ps::Player::One,
    const ps::RulesConfig &config = {}) {
  ps::GameState state = ps::make_initial_state(config);
  state.ball = point;
  state.to_move = to_move;
  state.status = ps::Status::InProgress;
  state.path = {point};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[point] = 1;
  return state;
}

bool same_state(const ps::GameState &lhs, const ps::GameState &rhs) {
  if (!(lhs.ball == rhs.ball && lhs.to_move == rhs.to_move && lhs.status == rhs.status &&
        lhs.path == rhs.path && lhs.config.width == rhs.config.width &&
        lhs.config.height == rhs.config.height &&
        lhs.config.goal_rule == rhs.config.goal_rule &&
        lhs.config.blocked_rule == rhs.config.blocked_rule)) {
    return false;
  }

  if (lhs.used_segments.size() != rhs.used_segments.size()) {
    return false;
  }
  for (const auto &segment : lhs.used_segments) {
    if (rhs.used_segments.find(segment) == rhs.used_segments.end()) {
      return false;
    }
  }

  if (lhs.visit_count.size() != rhs.visit_count.size()) {
    return false;
  }
  for (const auto &[point, count] : lhs.visit_count) {
    const auto it = rhs.visit_count.find(point);
    if (it == rhs.visit_count.end() || it->second != count) {
      return false;
    }
  }

  return true;
}

void initial_state_has_8_legal_moves_from_center() {
  const ps::GameState state = ps::make_initial_state();
  const auto moves = ps::legal_moves(state);
  require(state.ball == ps::Point{4, 6}, "Initial ball should be at (4, 6).");
  require(moves.size() == 8, "Initial state should have exactly 8 legal moves.");
}

void legal_moves_never_use_negative_coordinates() {
  const std::vector<ps::GameState> states{
      ps::make_initial_state(),
      make_clean_state_at(ps::Point{4, 1}, ps::Player::One),
      make_clean_state_at(ps::Point{4, 11}, ps::Player::Two),
  };

  for (const ps::GameState &state : states) {
    for (const ps::Move &move : ps::legal_moves(state)) {
      require(move.to.x >= 0 && move.to.y >= 0,
              "Legal move coordinates should never be negative.");
    }
  }
}

void reusing_segment_is_illegal_in_both_directions() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{{4, 5}});

  const auto moves = ps::legal_moves(state);
  require(!contains_move(moves, ps::Point{4, 6}),
          "Reverse move on already used segment should be illegal.");

  bool threw = false;
  try {
    (void)ps::apply_move(state, ps::Move{{4, 6}});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "Applying a reverse move over used segment must throw.");
}

void movement_along_outer_boundary_is_illegal() {
  ps::GameState state = make_clean_state_at(ps::Point{0, 6});
  const auto moves = ps::legal_moves(state);

  require(!contains_move(moves, ps::Point{0, 5}), "Left wall boundary segment must be illegal.");
  require(!contains_move(moves, ps::Point{0, 7}), "Left wall boundary segment must be illegal.");
}

void reaching_visited_point_grants_extra_turn() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{{4, 5}});  // P1
  state = ps::apply_move(state, ps::Move{{5, 5}});  // P2
  state = ps::apply_move(state, ps::Move{{5, 6}});  // P1

  require(state.to_move == ps::Player::Two, "Expected Player 2 before the visited-point move.");
  state = ps::apply_move(state, ps::Move{{4, 6}});  // P2 -> visited starting point
  require(state.to_move == ps::Player::Two,
          "Landing on previously visited point must grant an extra turn.");
}

void reaching_boundary_point_grants_extra_turn() {
  ps::GameState state = make_clean_state_at(ps::Point{2, 2}, ps::Player::One);
  state = ps::apply_move(state, ps::Move{{2, 1}});
  require(state.to_move == ps::Player::One,
          "Landing on boundary point must grant an extra turn.");
}

void reaching_middle_of_goal_mouth_does_not_grant_extra_turn() {
  ps::GameState north = make_clean_state_at(ps::Point{4, 2}, ps::Player::One);
  north = ps::apply_move(north, ps::Move{{4, 1}});
  require(north.to_move == ps::Player::Two,
          "The open center of the north goal mouth must not grant an extra turn.");

  ps::GameState south = make_clean_state_at(ps::Point{4, 10}, ps::Player::Two);
  south = ps::apply_move(south, ps::Move{{4, 11}});
  require(south.to_move == ps::Player::One,
          "The open center of the south goal mouth must not grant an extra turn.");
}

void goal_entry_sets_terminal_and_winner() {
  ps::GameState state = make_clean_state_at(ps::Point{4, 1}, ps::Player::One);
  require(contains_move(ps::legal_moves(state), ps::Point{4, 0}),
          "North goal entry should be legal from north center mouth.");

  state = ps::apply_move(state, ps::Move{{4, 0}});
  require(ps::is_terminal(state), "Entering the goal must end the game.");
  require(ps::winner(state).has_value() && ps::winner(state).value() == ps::Player::One,
          "Player 1 should win after entering north goal.");
}

void south_goal_entry_sets_terminal_and_winner() {
  ps::GameState state = make_clean_state_at(ps::Point{4, 11}, ps::Player::Two);
  require(contains_move(ps::legal_moves(state), ps::Point{4, 12}),
          "South goal entry should be legal from south center mouth.");

  state = ps::apply_move(state, ps::Move{{4, 12}});
  require(ps::is_terminal(state), "Entering the south goal must end the game.");
  require(ps::winner(state).has_value() && ps::winner(state).value() == ps::Player::Two,
          "Player 2 should win after entering south goal.");
}

void codingame_goal_rule_allows_own_goals_and_awards_the_goal_side() {
  const ps::GameState default_north =
      make_clean_state_at({4, 1}, ps::Player::Two);
  require(!contains_move(ps::legal_moves(default_north), {4, 0}),
          "The default rules should continue to hide a player's own goal.");

  ps::RulesConfig config;
  config.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  config.blocked_rule = ps::BlockedRule::MoverLoses;

  ps::GameState north = make_clean_state_at({4, 1}, ps::Player::Two, config);
  require(contains_move(ps::legal_moves(north), {4, 0}),
          "CodinGame rules should allow Player Two to enter the north goal.");
  north = ps::apply_move(north, ps::Move{{4, 0}});
  require(ps::winner(north) == ps::Player::One,
          "A north own goal should award the game to Player One.");

  ps::GameState south = make_clean_state_at({4, 11}, ps::Player::One, config);
  require(contains_move(ps::legal_moves(south), {4, 12}),
          "CodinGame rules should allow Player One to enter the south goal.");
  south = ps::apply_move(south, ps::Move{{4, 12}});
  require(ps::winner(south) == ps::Player::Two,
          "A south own goal should award the game to Player Two.");
}

void codingame_goal_posts_are_preserved_for_both_players() {
  ps::RulesConfig config;
  config.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  config.blocked_rule = ps::BlockedRule::MoverLoses;

  for (const ps::Player player : {ps::Player::One, ps::Player::Two}) {
    for (const std::pair<int, int> rows :
         {std::pair{1, 0}, std::pair{11, 12}}) {
      require(goal_destinations(make_clean_state_at({3, rows.first}, player,
                                                    config)) ==
                  std::vector<ps::Point>{{4, rows.second}},
              "A side mouth should only enter diagonally past its goal post.");
      require(goal_destinations(make_clean_state_at({4, rows.first}, player,
                                                    config)) ==
                  (std::vector<ps::Point>{{3, rows.second},
                                          {4, rows.second},
                                          {5, rows.second}}),
              "The center mouth should expose all three goal entries.");
      require(goal_destinations(make_clean_state_at({5, rows.first}, player,
                                                    config)) ==
                  std::vector<ps::Point>{{4, rows.second}},
              "The other side mouth should only enter diagonally past its goal post.");
    }
  }
}

void no_legal_moves_on_turn_causes_loss() {
  ps::GameState state = make_clean_state_at(ps::Point{4, 6}, ps::Player::One);
  const ps::Point trap_center{4, 5};
  const std::vector<ps::Point> blockers{
      {3, 4}, {4, 4}, {5, 4}, {3, 5}, {5, 5}, {3, 6}, {5, 6},
  };

  for (const ps::Point neighbor : blockers) {
    state.used_segments.insert(ps::Segment{trap_center, neighbor});
  }

  state = ps::apply_move(state, ps::Move{{4, 5}});
  require(state.status == ps::Status::WonByOne,
          "If next player has zero legal moves at turn start, they must lose.");
  require(ps::winner(state) == ps::Player::One,
          "Winner should be the player who made the trapping move.");
}

void codingame_blocked_landing_causes_the_mover_to_lose() {
  ps::RulesConfig config;
  config.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  config.blocked_rule = ps::BlockedRule::MoverLoses;
  ps::GameState state =
      make_clean_state_at({4, 6}, ps::Player::One, config);
  const ps::Point blocked{4, 5};
  const std::vector<ps::Point> blockers{
      {3, 4}, {4, 4}, {5, 4}, {3, 5}, {5, 5}, {3, 6}, {5, 6},
  };
  for (const ps::Point neighbor : blockers) {
    state.used_segments.insert(ps::Segment{blocked, neighbor});
  }

  state = ps::apply_move(state, ps::Move{blocked});
  require(ps::winner(state) == ps::Player::Two,
          "CodinGame should award a blocked landing to the mover's opponent.");
}

void goal_move_is_legal_only_from_goal_mouth_points() {
  ps::GameState not_mouth = make_clean_state_at(ps::Point{2, 1}, ps::Player::One);
  require(!contains_move(ps::legal_moves(not_mouth), ps::Point{3, 0}),
          "North goal should not be reachable from non-mouth point.");
  require(!contains_move(ps::legal_moves(not_mouth), ps::Point{4, 0}),
          "North goal should not be reachable from non-mouth point.");

  ps::GameState left_mouth = make_clean_state_at(ps::Point{3, 1}, ps::Player::One);
  require(!contains_move(ps::legal_moves(left_mouth), ps::Point{3, 0}),
          "Left mouth point should not move straight onto the left goal post.");
  require(contains_move(ps::legal_moves(left_mouth), ps::Point{4, 0}),
          "Left mouth point should connect diagonally to center goal point.");
  require(!contains_move(ps::legal_moves(left_mouth), ps::Point{5, 0}),
          "Left mouth point should not connect across two columns into the far goal point.");

  ps::GameState center_mouth = make_clean_state_at(ps::Point{4, 1}, ps::Player::One);
  require(contains_move(ps::legal_moves(center_mouth), ps::Point{3, 0}),
          "Center mouth point should connect diagonally to left goal point.");
  require(contains_move(ps::legal_moves(center_mouth), ps::Point{4, 0}),
          "Center mouth point should connect to center goal point.");
  require(contains_move(ps::legal_moves(center_mouth), ps::Point{5, 0}),
          "Center mouth point should connect diagonally to right goal point.");

  ps::GameState right_mouth = make_clean_state_at(ps::Point{5, 1}, ps::Player::One);
  require(contains_move(ps::legal_moves(right_mouth), ps::Point{4, 0}),
          "Right mouth point should connect diagonally to center goal point.");
  require(!contains_move(ps::legal_moves(right_mouth), ps::Point{5, 0}),
          "Right mouth point should not move straight onto the right goal post.");
}

void apply_move_is_pure_on_invalid_move() {
  ps::GameState state = ps::make_initial_state();
  const ps::GameState snapshot = state;

  bool threw = false;
  try {
    (void)ps::apply_move(state, ps::Move{{8, 8}});
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  require(threw, "Invalid move must throw.");
  require(same_state(state, snapshot), "apply_move must not mutate input state on failure.");
}

void renderer_includes_ball_goals_and_walls() {
  const ps::GameState state = ps::make_initial_state();
  const std::string rendered = ps::render_ascii(state);
  require(rendered.find("@") != std::string::npos, "Renderer should show current ball.");
  require(rendered.find("^") != std::string::npos, "Renderer should show north goal.");
  require(rendered.find("v") != std::string::npos, "Renderer should show south goal.");
  require(rendered.find("=") != std::string::npos, "Renderer should show top/bottom walls.");
  require(rendered.find("!") != std::string::npos, "Renderer should show left/right walls.");
}

void renderer_uses_zero_based_row_labels() {
  const std::string rendered = ps::render_ascii(ps::make_initial_state());
  const std::size_t board_start = rendered.find("\n\n");

  require(board_start != std::string::npos, "Renderer should separate metadata from the board.");
  require(rendered.compare(board_start + 2, 4, "  0 ") == 0,
          "The first rendered board row should be row 0.");
  require(rendered.find("\n -1 ") == std::string::npos,
          "Renderer should not include a negative row label.");
}

void renderer_shows_used_segments() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{{5, 6}});
  const std::string rendered = ps::render_ascii(state);
  require(rendered.find("-") != std::string::npos,
          "Renderer should include segment markers for used edges.");
}

}  // namespace

int run_rules_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };

  const std::vector<TestCase> tests{
      {"initial_state_has_8_legal_moves_from_center",
       initial_state_has_8_legal_moves_from_center},
      {"legal_moves_never_use_negative_coordinates",
       legal_moves_never_use_negative_coordinates},
      {"reusing_segment_is_illegal_in_both_directions",
       reusing_segment_is_illegal_in_both_directions},
      {"movement_along_outer_boundary_is_illegal", movement_along_outer_boundary_is_illegal},
      {"reaching_visited_point_grants_extra_turn", reaching_visited_point_grants_extra_turn},
      {"reaching_boundary_point_grants_extra_turn", reaching_boundary_point_grants_extra_turn},
      {"reaching_middle_of_goal_mouth_does_not_grant_extra_turn",
       reaching_middle_of_goal_mouth_does_not_grant_extra_turn},
      {"goal_entry_sets_terminal_and_winner", goal_entry_sets_terminal_and_winner},
      {"south_goal_entry_sets_terminal_and_winner",
       south_goal_entry_sets_terminal_and_winner},
      {"codingame_goal_rule_allows_own_goals_and_awards_the_goal_side",
       codingame_goal_rule_allows_own_goals_and_awards_the_goal_side},
      {"codingame_goal_posts_are_preserved_for_both_players",
       codingame_goal_posts_are_preserved_for_both_players},
      {"no_legal_moves_on_turn_causes_loss", no_legal_moves_on_turn_causes_loss},
      {"codingame_blocked_landing_causes_the_mover_to_lose",
       codingame_blocked_landing_causes_the_mover_to_lose},
      {"goal_move_is_legal_only_from_goal_mouth_points",
       goal_move_is_legal_only_from_goal_mouth_points},
      {"apply_move_is_pure_on_invalid_move", apply_move_is_pure_on_invalid_move},
      {"renderer_includes_ball_goals_and_walls", renderer_includes_ball_goals_and_walls},
      {"renderer_uses_zero_based_row_labels", renderer_uses_zero_based_row_labels},
      {"renderer_shows_used_segments", renderer_shows_used_segments},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << "\n";
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << "\n";
    } catch (...) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": unknown error\n";
    }
  }

  std::cout << "\n" << (tests.size() - static_cast<std::size_t>(failures)) << "/"
            << tests.size() << " tests passed.\n";
  return failures == 0 ? 0 : 1;
}
