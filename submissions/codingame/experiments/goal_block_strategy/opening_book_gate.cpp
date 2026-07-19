#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "../../paper_soccer_alpha_beta.cpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

struct Policy {
  bool goal_block_book{};
  bool first_execution{true};
  int replay_value_blend_percent{};
};

bool lookup_goal_block_opening(int player_id, std::string_view transcript,
                               std::string &action) {
  struct Entry {
    int player_id;
    std::string_view transcript;
    std::string_view action;
  };

  // This is the opening prefix from a public game in which the second-ranked
  // bot defeated the leader's current agent as Player 1.
  static constexpr Entry entries[] = {
      {1, "0", "2"},
      {1, "0/2/57", "6"},
      {1, "0/2/57/6/1", "7"},
      {1, "0/2/57/6/1/7/6", "3"},
      {1, "0/2/57/6/1/7/6/3/5", "3"},
      {1, "0/2/57/6/1/7/6/3/5/3/001", "3"},
      {1, "0/2/57/6/1/7/6/3/5/3/001/3/606430300", "3"},
      {1, "0/2/57/6/1/7/6/3/5/3/001/3/606430300/3/561300", "521"},
      {1, "0/2/57/6/1/7/6/3/5/3/001/3/606430300/3/561300/521/660", "33"},
      {1, "0/2/57/6/1/7/6/3/5/3/001/3/606430300/3/561300/521/660/33/00", "7574"},
  };

  for (const Entry &entry : entries) {
    if (entry.player_id == player_id && entry.transcript == transcript) {
      action = entry.action;
      return true;
    }
  }
  return false;
}

std::string choose(Policy &policy, ps::GameState &state,
                   std::string_view transcript, std::uint32_t first_ms,
                   std::uint32_t later_ms, std::uint64_t node_budget) {
  const int player_id = state.to_move == ps::Player::One ? 0 : 1;
  std::string action;
  if (policy.goal_block_book &&
      lookup_goal_block_opening(player_id, transcript, action)) {
    cg::apply_encoded_turn(state, action);
  } else if (!cg::try_replay_correction(state, player_id, transcript, action)) {
    cg::SearchConfig config;
    config.replay_value_blend_percent = policy.replay_value_blend_percent;
    if (node_budget != 0) {
      config.max_nodes = node_budget;
      config.max_time_ms = 0;
    } else {
      const std::uint32_t search_time_ms =
          policy.first_execution ? first_ms : later_ms;
      config.absolute_deadline =
          cg::SearchClock::now() + std::chrono::milliseconds(search_time_ms);
    }
    cg::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("search returned an overlong action");
      }
      action.push_back(cg::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    if (action.empty() || (!ps::is_terminal(state) && state.to_move == mover)) {
      throw std::logic_error("search returned an incomplete action");
    }
  }
  policy.first_execution = false;
  return action;
}

struct Result {
  int winner{-1};
  int turns{};
  std::string transcript;
};

Result play(bool candidate_is_player_zero, std::uint32_t first_ms,
            std::uint32_t later_ms, std::uint64_t node_budget,
            int maximum_turns, bool verbose, bool book_enabled,
            int candidate_replay_value_blend_percent) {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  ps::GameState state = ps::make_initial_state(rules);
  Policy player_zero{book_enabled && candidate_is_player_zero};
  Policy player_one{book_enabled && !candidate_is_player_zero};
  if (candidate_is_player_zero) {
    player_zero.replay_value_blend_percent =
        candidate_replay_value_blend_percent;
  } else {
    player_one.replay_value_blend_percent =
        candidate_replay_value_blend_percent;
  }
  std::string transcript;

  int turns = 0;
  while (!ps::is_terminal(state) && turns < maximum_turns) {
    const int mover = state.to_move == ps::Player::One ? 0 : 1;
    Policy &policy = mover == 0 ? player_zero : player_one;
    const std::string action =
        choose(policy, state, transcript, first_ms, later_ms, node_budget);
    if (!transcript.empty()) {
      transcript.push_back('/');
    }
    transcript += action;
    if (verbose) {
      std::cout << turns << ' ' << mover << ' ' << action << ' ' << state.ball.x
                << ' ' << state.ball.y << '\n';
    }
    ++turns;
  }

  int winning_player = -1;
  if (const auto winning_side = ps::winner(state)) {
    winning_player = *winning_side == ps::Player::One ? 0 : 1;
  }
  return {winning_player, turns, std::move(transcript)};
}

}  // namespace

int main(int argc, char **argv) {
  const int games_per_color = argc > 1 ? std::atoi(argv[1]) : 1;
  const std::uint32_t first_ms =
      argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 650U;
  const std::uint32_t later_ms =
      argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 130U;
  const int maximum_turns = argc > 4 ? std::atoi(argv[4]) : 300;
  const std::uint64_t node_budget =
      argc > 5 ? static_cast<std::uint64_t>(std::strtoull(argv[5], nullptr, 10))
               : 0U;
  const bool book_enabled = argc <= 6 || std::string_view(argv[6]) != "control";
  const int candidate_replay_value_blend_percent =
      argc > 7 ? std::atoi(argv[7]) : 0;
  const bool verbose = argc > 8 && std::string_view(argv[8]) == "--verbose";

  int candidate_wins = 0;
  int baseline_wins = 0;
  int unfinished = 0;
  for (int color = 0; color < 2; ++color) {
    const bool candidate_is_player_zero = color == 0;
    for (int game = 0; game < games_per_color; ++game) {
      const Result result =
          play(candidate_is_player_zero, first_ms, later_ms, node_budget,
               maximum_turns, verbose, book_enabled,
               candidate_replay_value_blend_percent);
      const int candidate_player = candidate_is_player_zero ? 0 : 1;
      if (result.winner == candidate_player) {
        ++candidate_wins;
      } else if (result.winner >= 0) {
        ++baseline_wins;
      } else {
        ++unfinished;
      }
      std::cout << "game color=" << candidate_player << " winner="
                << result.winner << " turns=" << result.turns
                << " transcript=" << result.transcript << '\n';
    }
  }
  std::cout << "summary candidate=" << candidate_wins
            << " baseline=" << baseline_wins << " unfinished=" << unfinished
            << '\n';
  return unfinished == 0 ? 0 : 2;
}
