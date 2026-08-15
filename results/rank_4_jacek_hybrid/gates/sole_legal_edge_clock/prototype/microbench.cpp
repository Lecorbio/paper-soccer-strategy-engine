#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "bot.cpp"
#undef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

ps::GameState reconstruct(std::string_view transcript) {
  ps::RulesConfig rules{8, 10, ps::GoalRule::OwnGoalsAllowed,
                        ps::BlockedRule::MoverLoses};
  ps::GameState state = ps::make_initial_state(rules);
  if (transcript.empty()) return state;
  std::size_t begin = 0;
  while (begin <= transcript.size()) {
    const std::size_t slash = transcript.find('/', begin);
    const std::size_t end = slash == std::string_view::npos
                                ? transcript.size() : slash;
    cg::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (slash == std::string_view::npos) break;
    begin = slash + 1;
  }
  if (ps::is_terminal(state)) throw std::logic_error("terminal fixture");
  return state;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const std::string_view panel(argv[1]);
  const bool forced = panel.starts_with("forced");
  const bool production = panel.ends_with("prod");
  const std::string_view forced_transcript =
      "7/7/1/6/6/61/2/2/3/1/13/4/36/4/36/36/6/0/5/74/4/2/71/0/56/"
      "3244/4/7/06/4/4/41/25/013/0/5250302/0/5302761/6/65617/10/21";
  ps::GameState state = reconstruct(
      forced ? forced_transcript : "4/3/6/4/3/0");
  cg::SearchConfig config;
  config.max_nodes = 50'000;
  if (!production) {
    config.transposition_entries = 4096;
    config.evaluation_entries = 2048;
  }
  config.exact_proof_mask = 7;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  const auto begin = std::chrono::steady_clock::now();
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  const auto end = std::chrono::steady_clock::now();
  const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         end - begin).count();
  const cg::SearchStats &s = search.stats();
  std::cout << nanos << ' ' << s.nodes << ' ' << s.root_score << ' '
            << action.size() << ' ' << s.forced_edges << ' '
            << s.completed_turn_depth << ' ' << s.transposition_hits << '\n';
}
