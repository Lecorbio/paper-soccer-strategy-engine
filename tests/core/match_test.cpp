#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "papersoccer/match.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool same_state(const ps::GameState &lhs, const ps::GameState &rhs) {
  if (!(lhs.config.width == rhs.config.width && lhs.config.height == rhs.config.height &&
        lhs.config.goal_rule == rhs.config.goal_rule &&
        lhs.config.blocked_rule == rhs.config.blocked_rule &&
        lhs.ball == rhs.ball && lhs.to_move == rhs.to_move && lhs.status == rhs.status &&
        lhs.path == rhs.path && lhs.used_segments.size() == rhs.used_segments.size() &&
        lhs.visit_count.size() == rhs.visit_count.size())) {
    return false;
  }

  for (const ps::Segment &segment : lhs.used_segments) {
    if (rhs.used_segments.find(segment) == rhs.used_segments.end()) {
      return false;
    }
  }
  for (const auto &[point, count] : lhs.visit_count) {
    const auto it = rhs.visit_count.find(point);
    if (it == rhs.visit_count.end() || it->second != count) {
      return false;
    }
  }
  return true;
}

void new_match_owns_an_initial_state_and_empty_history() {
  const ps::Match match;

  require(match.state().ball == ps::Point{4, 6},
          "A match should start at the rules engine's initial point.");
  require(match.state().to_move == ps::Player::One,
          "Player 1 should own the initial turn.");
  require(match.history().empty(), "A new match should have no played moves.");
  require(match.legal_moves().size() == 8,
          "A new match should expose all eight initial legal moves.");
}

void play_updates_state_and_records_authoritative_move_metadata() {
  ps::Match match;

  const ps::PlayedMove played = match.play(ps::Move{{4, 5}});

  require(played.ply == 1, "The first played move should be ply 1.");
  require(played.player == ps::Player::One, "The move should record its acting player.");
  require(played.from == ps::Point{4, 6} && played.to == ps::Point{4, 5},
          "The move should record its exact endpoints.");
  require(!played.extra_turn, "The first move should not be marked as a rebound.");
  require(played.status_after == ps::Status::InProgress,
          "The first move should leave the match in progress.");
  require(match.history().size() == 1 && match.history().front() == played,
          "The returned metadata should be the metadata retained by the match.");
  require(match.state().ball == ps::Point{4, 5} &&
              match.state().to_move == ps::Player::Two,
          "The owned state should advance after a successful move.");
}

void played_move_records_rebounds_before_the_state_changes() {
  ps::Match match;
  (void)match.play(ps::Move{{4, 5}});
  (void)match.play(ps::Move{{5, 5}});
  (void)match.play(ps::Move{{5, 6}});

  const ps::PlayedMove rebound = match.play(ps::Move{{4, 6}});

  require(rebound.player == ps::Player::Two,
          "The rebound should retain the player who made the move.");
  require(rebound.extra_turn,
          "Returning to the visited starting point should be recorded as an extra turn.");
  require(match.state().to_move == ps::Player::Two,
          "The player should retain the turn after the recorded rebound.");
}

void rejected_move_does_not_change_state_or_history() {
  ps::Match match;
  const ps::GameState state_before = match.state();
  const std::vector<ps::PlayedMove> history_before = match.history();

  bool threw = false;
  try {
    (void)match.play(ps::Move{{8, 8}});
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  require(threw, "Match::play should retain the rules engine's illegal-move rejection.");
  require(same_state(match.state(), state_before),
          "A rejected move should leave the owned game state unchanged.");
  require(match.history() == history_before,
          "A rejected move should leave the move history unchanged.");
}

void match_respects_custom_rules_configuration() {
  const ps::Match match(ps::RulesConfig{10, 12});

  require(match.state().config.width == 10 && match.state().config.height == 12,
          "The match should retain its supplied rules configuration.");
  require(match.state().ball == ps::Point{5, 7},
          "The initial point should derive from the supplied rules configuration.");
}

void undo_restores_each_prior_authoritative_state() {
  ps::Match match;
  std::vector<ps::GameState> states{match.state()};
  const std::vector<ps::Move> moves{
      ps::Move{{4, 5}},
      ps::Move{{5, 5}},
      ps::Move{{5, 6}},
      ps::Move{{4, 6}},
  };

  for (const ps::Move move : moves) {
    (void)match.play(move);
    states.push_back(match.state());
  }
  require(match.history().back().extra_turn,
          "The undo fixture should include a rebound onto a visited point.");

  for (std::size_t remaining = moves.size(); remaining > 0; --remaining) {
    const std::optional<ps::PlayedMove> undone = match.undo();
    require(undone.has_value() && undone->ply == remaining,
            "Undo should return and remove exactly the latest played move.");
    require(match.history().size() == remaining - 1,
            "Undo should shorten the authoritative history by one ply.");
    require(same_state(match.state(), states[remaining - 1]),
            "Undo should restore the complete state from before that ply.");
  }

  require(!match.undo().has_value(),
          "Undo at the initial position should be a non-mutating no-op.");
  require(same_state(match.state(), states.front()) && match.history().empty(),
          "Repeated undo should stop at the exact initial state.");
}

void undo_reopens_a_terminal_match() {
  ps::Match match(ps::RulesConfig{8, 1});
  const ps::GameState initial = match.state();

  const ps::PlayedMove goal = match.play(ps::Move{{4, 0}});
  require(goal.status_after == ps::Status::WonByOne &&
              ps::is_terminal(match.state()),
          "The compact-board fixture should finish with an immediate goal.");

  const std::optional<ps::PlayedMove> undone = match.undo();
  require(undone == goal && same_state(match.state(), initial) &&
              match.legal_moves().size() > 0,
          "Undoing a winning move should restore the playable pre-goal state.");
}

}  // namespace

int run_match_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };

  const std::vector<TestCase> tests{
      {"new_match_owns_an_initial_state_and_empty_history",
       new_match_owns_an_initial_state_and_empty_history},
      {"play_updates_state_and_records_authoritative_move_metadata",
       play_updates_state_and_records_authoritative_move_metadata},
      {"played_move_records_rebounds_before_the_state_changes",
       played_move_records_rebounds_before_the_state_changes},
      {"rejected_move_does_not_change_state_or_history",
       rejected_move_does_not_change_state_or_history},
      {"match_respects_custom_rules_configuration",
       match_respects_custom_rules_configuration},
      {"undo_restores_each_prior_authoritative_state",
       undo_restores_each_prior_authoritative_state},
      {"undo_reopens_a_terminal_match", undo_reopens_a_terminal_match},
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
            << tests.size() << " match tests passed.\n";
  return failures == 0 ? 0 : 1;
}

#ifdef PAPERSOCCER_STANDALONE_MATCH_TEST_MAIN
int main() { return run_match_tests(); }
#endif
