#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "papersoccer/debug.hpp"
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

ps::GameState make_clean_state_at(ps::Point point, ps::Player to_move = ps::Player::One) {
  ps::GameState state = ps::make_initial_state();
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
        lhs.config.height == rhs.config.height)) {
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
  require(moves.size() == 8, "Initial state should have exactly 8 legal moves.");
}

void reusing_segment_is_illegal_in_both_directions() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{{4, 4}});

  const auto moves = ps::legal_moves(state);
  require(!contains_move(moves, ps::Point{4, 5}),
          "Reverse move on already used segment should be illegal.");

  bool threw = false;
  try {
    (void)ps::apply_move(state, ps::Move{{4, 5}});
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, "Applying a reverse move over used segment must throw.");
}

void movement_along_outer_boundary_is_illegal() {
  ps::GameState state = make_clean_state_at(ps::Point{0, 5});
  const auto moves = ps::legal_moves(state);

  require(!contains_move(moves, ps::Point{0, 4}), "Left wall boundary segment must be illegal.");
  require(!contains_move(moves, ps::Point{0, 6}), "Left wall boundary segment must be illegal.");
}

void reaching_visited_point_grants_extra_turn() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{{4, 4}});  // P1
  state = ps::apply_move(state, ps::Move{{5, 4}});  // P2
  state = ps::apply_move(state, ps::Move{{5, 5}});  // P1

  require(state.to_move == ps::Player::Two, "Expected Player 2 before the visited-point move.");
  state = ps::apply_move(state, ps::Move{{4, 5}});  // P2 -> visited starting point
  require(state.to_move == ps::Player::Two,
          "Landing on previously visited point must grant an extra turn.");
}

void reaching_boundary_point_grants_extra_turn() {
  ps::GameState state = make_clean_state_at(ps::Point{4, 1}, ps::Player::One);
  state = ps::apply_move(state, ps::Move{{4, 0}});
  require(state.to_move == ps::Player::One,
          "Landing on boundary point must grant an extra turn.");
}

void goal_entry_sets_terminal_and_winner() {
  ps::GameState state = make_clean_state_at(ps::Point{4, 0}, ps::Player::One);
  require(contains_move(ps::legal_moves(state), ps::Point{4, -1}),
          "North goal entry should be legal from north center mouth.");

  state = ps::apply_move(state, ps::Move{{4, -1}});
  require(ps::is_terminal(state), "Entering the goal must end the game.");
  require(ps::winner(state).has_value() && ps::winner(state).value() == ps::Player::One,
          "Player 1 should win after entering north goal.");
}

void no_legal_moves_on_turn_causes_loss() {
  ps::GameState state = make_clean_state_at(ps::Point{4, 5}, ps::Player::One);
  const ps::Point trap_center{4, 4};
  const std::vector<ps::Point> blockers{
      {3, 3}, {4, 3}, {5, 3}, {3, 4}, {5, 4}, {3, 5}, {5, 5},
  };

  for (const ps::Point neighbor : blockers) {
    state.used_segments.insert(ps::Segment{trap_center, neighbor});
  }

  state = ps::apply_move(state, ps::Move{{4, 4}});
  require(state.status == ps::Status::WonByOne,
          "If next player has zero legal moves at turn start, they must lose.");
  require(ps::winner(state).has_value() && ps::winner(state).value() == ps::Player::One,
          "Winner should be the player who made the trapping move.");
}

void goal_move_is_legal_only_from_goal_mouth_points() {
  ps::GameState not_mouth = make_clean_state_at(ps::Point{2, 0}, ps::Player::One);
  require(!contains_move(ps::legal_moves(not_mouth), ps::Point{3, -1}),
          "North goal should not be reachable from non-mouth point.");
  require(!contains_move(ps::legal_moves(not_mouth), ps::Point{4, -1}),
          "North goal should not be reachable from non-mouth point.");

  ps::GameState left_mouth = make_clean_state_at(ps::Point{3, 0}, ps::Player::One);
  require(!contains_move(ps::legal_moves(left_mouth), ps::Point{3, -1}),
          "Left mouth point should not move straight onto the left goal post.");
  require(contains_move(ps::legal_moves(left_mouth), ps::Point{4, -1}),
          "Left mouth point should connect diagonally to center goal point.");
  require(!contains_move(ps::legal_moves(left_mouth), ps::Point{5, -1}),
          "Left mouth point should not connect across two columns into the far goal point.");

  ps::GameState center_mouth = make_clean_state_at(ps::Point{4, 0}, ps::Player::One);
  require(contains_move(ps::legal_moves(center_mouth), ps::Point{3, -1}),
          "Center mouth point should connect diagonally to left goal point.");
  require(contains_move(ps::legal_moves(center_mouth), ps::Point{4, -1}),
          "Center mouth point should connect to center goal point.");
  require(contains_move(ps::legal_moves(center_mouth), ps::Point{5, -1}),
          "Center mouth point should connect diagonally to right goal point.");

  ps::GameState right_mouth = make_clean_state_at(ps::Point{5, 0}, ps::Player::One);
  require(contains_move(ps::legal_moves(right_mouth), ps::Point{4, -1}),
          "Right mouth point should connect diagonally to center goal point.");
  require(!contains_move(ps::legal_moves(right_mouth), ps::Point{5, -1}),
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

void renderer_shows_used_segments() {
  ps::GameState state = ps::make_initial_state();
  state = ps::apply_move(state, ps::Move{{5, 5}});
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
      {"reusing_segment_is_illegal_in_both_directions",
       reusing_segment_is_illegal_in_both_directions},
      {"movement_along_outer_boundary_is_illegal", movement_along_outer_boundary_is_illegal},
      {"reaching_visited_point_grants_extra_turn", reaching_visited_point_grants_extra_turn},
      {"reaching_boundary_point_grants_extra_turn", reaching_boundary_point_grants_extra_turn},
      {"goal_entry_sets_terminal_and_winner", goal_entry_sets_terminal_and_winner},
      {"no_legal_moves_on_turn_causes_loss", no_legal_moves_on_turn_causes_loss},
      {"goal_move_is_legal_only_from_goal_mouth_points",
       goal_move_is_legal_only_from_goal_mouth_points},
      {"apply_move_is_pure_on_invalid_move", apply_move_is_pure_on_invalid_move},
      {"renderer_includes_ball_goals_and_walls", renderer_includes_ball_goals_and_walls},
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
