#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#ifndef FRONTIER_IMPL
#error "FRONTIER_IMPL must name the generated submission"
#endif
#include FRONTIER_IMPL

#include <array>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

ps::RulesConfig rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed,
          ps::BlockedRule::MoverLoses};
}

std::vector<std::string_view> split(std::string_view transcript) {
  std::vector<std::string_view> turns;
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t slash = transcript.find('/', begin);
    turns.push_back(transcript.substr(
        begin, slash == std::string_view::npos ? transcript.size() - begin
                                               : slash - begin));
    if (slash == std::string_view::npos) {
      break;
    }
    begin = slash + 1;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(rules());
  for (const std::string_view action : split(transcript)) {
    cg::apply_encoded_turn(state, action);
  }
  return state;
}

std::string encode(const ps::GameState &state,
                   const std::vector<ps::Move> &moves) {
  ps::GameState replay = state;
  std::string result;
  for (const ps::Move move : moves) {
    result.push_back(cg::encode_direction(replay.ball, move.to));
    replay = ps::apply_move(replay, move);
  }
  return result;
}

}  // namespace

int main() {
  constexpr std::array<std::string_view, 8> transcripts{{
      "",
      "0/6",
      "6/1",
      "0/6/5/4/5/53/61/0633",
      "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
      "0/2/7/45/7/5/71/34/2212/2/7/1/6/1/03636074/33535",
      "7/6/7/53/10/34/71/45/221/2/1/35/70/54/17/43/660/33",
      "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/532/27412/41/654",
  }};
  constexpr std::array<std::uint64_t, 3> budgets{{251, 2'000, 10'000}};
  std::cout << "case,budget,action,score,nodes,depth,attempted,leaf_probes,"
               "leaf_win,leaf_loss,eval_probes,eval_hits,tt_hits,tt_cutoffs\n";
  for (std::size_t index = 0; index < transcripts.size(); ++index) {
    const ps::GameState state = reconstruct(transcripts[index]);
    for (const std::uint64_t budget : budgets) {
      cg::SearchConfig config;
      config.max_nodes = budget;
      config.max_time_ms = 0;
      config.exact_proof_mask = cg::kOperationalExactProofs;
      config.replay_value_blend_percent = 15;
      config.teacher_residual_weight_percent = 100;
      cg::CompleteTurnSearch search(state, config);
      const std::vector<ps::Move> moves = search.run();
      const cg::SearchStats &stats = search.stats();
      std::cout << index << ',' << budget << ',' << encode(state, moves) << ','
                << stats.root_score << ',' << stats.nodes << ','
                << stats.completed_turn_depth << ','
                << stats.attempted_turn_depth << ','
                << stats.leaf_rebound_probes << ','
                << stats.leaf_rebound_win_hits << ','
                << stats.leaf_rebound_loss_hits << ','
                << stats.evaluation_cache_probes << ','
                << stats.evaluation_cache_hits << ','
                << stats.transposition_hits << ','
                << stats.transposition_cutoffs << '\n';
    }
  }
}
