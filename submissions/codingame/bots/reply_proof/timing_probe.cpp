#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "submission.cpp"

#include <chrono>
#include <array>
#include <iostream>
#include <string>
#include <string_view>

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
    cg::apply_encoded_turn(state, "0");
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

void apply_transcript(ps::GameState &state, std::string_view transcript) {
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    cg::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) break;
    begin = separator + 1U;
  }
}

double probe_reply_position(std::string_view transcript) {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  ps::GameState state = ps::make_initial_state(rules);
  apply_transcript(state, transcript);
  if (ps::is_terminal(state)) {
    throw std::logic_error("timing reply position is terminal");
  }
  return timed_choice(state, cg::kLaterSearchTimeMs);
}

}  // namespace

int main() {
  for (int player_id = 0; player_id < 2; ++player_id) {
    const auto [first, later] = probe(player_id);
    std::cout << "player=" << player_id << " first_ms=" << first
              << " later_ms=" << later << '\n';
  }
  constexpr std::array<std::pair<std::string_view, std::string_view>, 3>
      kReplyCases{{
          {"elite-d2",
           "1/7/7/2/1/13/5/74/5027/20555/0/6/61/052434/7/61453/11/"
           "063664/23/6/4/4/2/3"},
          {"rank1-d1",
           "0/0/3/2/7/46160/30/57/6/6/53/61/13/123/556/3/6/143/6/5/0/"
           "143/00/2/55/23"},
          {"elite-d0-dense",
           "5/6/3/3/2/3/50/30675/47/6/3165061/631/671/4/1671/3/"
           "3454707210/3/64121/4/1/3/17/4/4271617/035/075/5/6/3/32107/"
           "743/067"},
      }};
  for (const auto &[name, transcript] : kReplyCases) {
    std::cout << "shell=" << name
              << " later_ms=" << probe_reply_position(transcript) << '\n';
  }
  return 0;
}
