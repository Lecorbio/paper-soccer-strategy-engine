#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "paper_soccer_rank_one.cpp"

#include <chrono>
#include <iostream>
#include <string>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

double timed_choice(ps::GameState &state, std::uint32_t budget_ms) {
  const auto start = std::chrono::steady_clock::now();
  const std::string action = cg::choose_complete_turn(state, budget_ms);
  const auto finish = std::chrono::steady_clock::now();
  if (action.empty()) {
    throw std::logic_error("timing probe produced an empty action");
  }
  return std::chrono::duration<double, std::milli>(finish - start).count();
}

std::pair<double, double> probe(int player_id) {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  ps::GameState state = ps::make_initial_state(rules);
  if (player_id == 1) {
    (void)timed_choice(state, 1);
  }
  const double first = timed_choice(state, cg::kFirstSearchTimeMs);
  if (ps::is_terminal(state)) {
    throw std::logic_error("timing probe ended before a later response");
  }
  (void)timed_choice(state, 1);
  if (ps::is_terminal(state)) {
    throw std::logic_error("timing probe opponent ended the game");
  }
  const double later = timed_choice(state, cg::kLaterSearchTimeMs);
  return {first, later};
}

}  // namespace

int main() {
  bool valid = true;
  for (int player_id = 0; player_id < 2; ++player_id) {
    const auto [first, later] = probe(player_id);
    std::cout << "player=" << player_id << " first_ms=" << first
              << " later_ms=" << later << '\n';
    valid = valid && first < 1000.0 && later < 200.0;
  }
  return valid ? 0 : 1;
}
