#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

#define private public
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "bot.cpp"
#undef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#undef private

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;
namespace detail = papersoccer::detail;

namespace {

ps::RulesConfig rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed,
          ps::BlockedRule::MoverLoses};
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

std::vector<ps::GameState> corpus(std::size_t target) {
  std::vector<ps::GameState> states;
  states.reserve(target + 8);
  std::uint64_t random = 0x4f1bbcdc676f2b31ULL;
  while (states.size() < target) {
    ps::GameState state = ps::make_initial_state(rules());
    while (!ps::is_terminal(state) && states.size() < target) {
      states.push_back(state);
      const std::vector<ps::Move> legal = ps::legal_moves(state);
      if (legal.empty()) throw std::logic_error("live state has no move");
      state = ps::apply_move(state, legal[next_random(random) % legal.size()]);
    }
  }
  for (const std::string_view transcript : {
           "6/1", "4/3/6/4/3/0", "0/6/5/4/5/53/61/0633",
           "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35"}) {
    ps::GameState state = ps::make_initial_state(rules());
    std::size_t begin = 0;
    while (begin <= transcript.size()) {
      const std::size_t slash = transcript.find('/', begin);
      const std::size_t end = slash == std::string_view::npos
                                  ? transcript.size() : slash;
      cg::apply_encoded_turn(state, transcript.substr(begin, end - begin));
      if (slash == std::string_view::npos) break;
      begin = slash + 1;
    }
    if (!ps::is_terminal(state)) states.push_back(std::move(state));
  }
  return states;
}

std::string encode(ps::GameState state, const std::vector<ps::Move> &action) {
  std::string result;
  for (const ps::Move move : action) {
    result.push_back(cg::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  return result;
}

void print_stats(const cg::SearchStats &s) {
  std::cout
      << s.completed_turn_depth << ',' << s.attempted_turn_depth << ','
      << s.nodes << ',' << s.leaf_evaluations << ',' << s.terminal_nodes << ','
      << s.completed_actions << ',' << s.cutoffs << ','
      << s.transposition_probes << ',' << s.transposition_hits << ','
      << s.transposition_cutoffs << ',' << s.transposition_stores << ','
      << s.continuation_transposition_hits << ','
      << s.evaluation_cache_probes << ',' << s.evaluation_cache_hits << ','
      << s.teacher_residual_evaluations << ',' << s.terminal_bound_cutoffs << ','
      << s.rebound_goal_probes << ',' << s.rebound_goal_hits << ','
      << s.rebound_loss_hits << ',' << s.root_rebound_probes << ','
      << s.root_rebound_win_hits << ',' << s.root_rebound_loss_hits << ','
      << s.leaf_rebound_probes << ',' << s.leaf_rebound_win_hits << ','
      << s.leaf_rebound_loss_hits << ',' << s.exchange_ply1_probes << ','
      << s.exchange_ply1_win_hits << ',' << s.exchange_ply1_loss_hits << ','
      << s.exchange_ply1_cutoffs << ',' << s.exchange_ply2_probes << ','
      << s.exchange_ply2_win_hits << ',' << s.exchange_ply2_loss_hits << ','
      << s.exchange_ply2_cutoffs << ',' << s.forced_edges << ','
      << s.root_seed_actions << ',' << s.root_transposition_reuses << ','
      << s.max_action_edges << ',' << s.root_score << ','
      << static_cast<int>(s.budget_exhausted);
}

}  // namespace

int main() {
  const auto states = corpus(1000);
  std::size_t singleton = 0;
  std::size_t singleton_nonzero_score = 0;
  for (std::size_t index = 0; index < states.size(); ++index) {
    cg::SearchConfig inspect;
    inspect.max_nodes = 1;
    inspect.transposition_entries = 2;
    inspect.evaluation_entries = 2;
    cg::CompleteTurnSearch search(states[index], inspect);
    const cg::OrderedMoveList moves = search.ordered_moves();
    if (moves.count == 1) {
      ++singleton;
      singleton_nonzero_score += moves.values[0].score != 0;
      std::cout << "S " << index << ' ' << static_cast<unsigned>(moves.values[0].slot)
                << ' ' << moves.values[0].move.to.x << ' '
                << moves.values[0].move.to.y << '\n';
    }
  }
  std::cerr << "states=" << states.size() << " singleton=" << singleton
            << " singleton_nonzero_score=" << singleton_nonzero_score << '\n';
  if (singleton < 10) return 2;
#if defined(CANDIDATE_FAST_PATH)
  if (singleton_nonzero_score != 0) return 3;
#endif

  constexpr std::array<std::uint64_t, 5> budgets{{1, 16, 64, 256, 1024}};
  constexpr std::array<std::uint8_t, 2> masks{{0, 7}};
  for (std::size_t index = 0; index < states.size(); ++index) {
    for (const std::uint8_t mask : masks) {
      for (const std::uint64_t budget : budgets) {
        cg::SearchConfig config;
        config.max_nodes = budget;
        config.transposition_entries = 4096;
        config.evaluation_entries = 2048;
        config.exact_proof_mask = mask;
        config.replay_value_blend_percent = 15;
        config.teacher_residual_weight_percent = 100;
        cg::CompleteTurnSearch search(states[index], config);
        const std::vector<ps::Move> action = search.run();
        std::cout << "D " << index << ' ' << static_cast<unsigned>(mask) << ' '
                  << budget << ' ' << encode(states[index], action) << ' ';
        print_stats(search.stats());
        std::cout << '\n';
      }
    }
  }
}
