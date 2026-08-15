#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_gate_reference_engine
#include "../rank_4/bot.cpp"
#undef turn_action_v2

#include "comparison_gate_engine.hpp"

#include <chrono>

namespace papersoccer::rank4_jacek_gate {

EngineDecision choose_rank4(const GameState &state,
                            const EngineConfig &config) {
  namespace engine = papersoccer::rank4_gate_reference_engine;
  engine::SearchConfig search_config;
  search_config.max_nodes = config.max_nodes;
  search_config.replay_value_blend_percent = 15;
  search_config.teacher_residual_weight_percent = 100;
  if (config.time_ms != 0) {
    search_config.absolute_deadline =
        engine::SearchClock::now() +
        std::chrono::milliseconds(config.time_ms);
  }
  engine::CompleteTurnSearch search(state, search_config);
  EngineDecision decision;
  decision.moves = search.run();
  const engine::SearchStats &stats = search.stats();
  decision.nodes = stats.nodes;
  decision.completed_depth = stats.completed_turn_depth;
  decision.attempted_depth = stats.attempted_turn_depth;
  decision.budget_exhausted = stats.budget_exhausted;
  return decision;
}

}  // namespace papersoccer::rank4_jacek_gate
