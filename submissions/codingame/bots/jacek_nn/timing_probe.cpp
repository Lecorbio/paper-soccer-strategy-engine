#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "submission.cpp"

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

  ps::GameState later_state = ps::make_initial_state(rules);
  for (const std::string_view action :
       {"0", "6", "5", "4", "5", "53", "61", "0633"}) {
    cg::apply_encoded_turn(later_state, action);
  }
  if (player_id == 1) {
    cg::apply_encoded_turn(later_state, "5024701");
  }
  if (ps::is_terminal(later_state) ||
      static_cast<int>(later_state.to_move) != player_id ||
      later_state.used_segments.size() < 12) {
    throw std::logic_error("timing probe constructed the wrong later state");
  }
  const double later = timed_choice(later_state, cg::kLaterSearchTimeMs);
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
