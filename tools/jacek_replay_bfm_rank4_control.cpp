#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 jacek_replay_rank4_control_engine
#include "../submissions/codingame/bots/rank_4/bot.cpp"
#undef turn_action_v2

#include "jacek_replay_bfm_gate_internal.hpp"

#include <chrono>

namespace papersoccer::jacek_replay_gate {

ControlDecision choose_rank4_control(const GameState &state,
                                     const ControlConfig &config) {
  namespace engine = papersoccer::jacek_replay_rank4_control_engine;
  engine::SearchConfig search_config;
  search_config.max_nodes = config.work_cap;
  search_config.replay_value_blend_percent = 15;
  search_config.teacher_residual_weight_percent = 100;
  if (config.time_ms != 0) {
    search_config.absolute_deadline =
        engine::SearchClock::now() +
        std::chrono::milliseconds(config.time_ms);
  }
  engine::CompleteTurnSearch search(state, search_config);
  ControlDecision result;
  result.action = search.run();
  const engine::SearchStats &stats = search.stats();
  result.work = stats.nodes;
  result.depth = stats.completed_turn_depth;
  result.root_value = static_cast<float>(stats.root_score) / 100'000.0F;
  result.deadline_reached = stats.budget_exhausted;
  result.cap_reached = stats.nodes >= config.work_cap;
  return result;
}

}  // namespace papersoccer::jacek_replay_gate
