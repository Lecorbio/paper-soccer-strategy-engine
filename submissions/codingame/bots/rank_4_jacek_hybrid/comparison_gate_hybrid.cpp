#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_jacek_gate_candidate_engine
#include "bot.cpp"
#undef turn_action_v2

#include "comparison_gate_engine.hpp"

#include <chrono>

namespace papersoccer::rank4_jacek_gate {

EngineDecision choose_hybrid(const GameState &state,
                             const EngineConfig &config) {
  namespace engine = papersoccer::rank4_jacek_gate_candidate_engine;
  engine::SearchConfig search_config;
  search_config.max_nodes = config.max_nodes;
  search_config.replay_value_blend_percent = 15;
  search_config.teacher_residual_weight_percent = 100;
  search_config.exact_proof_mask = config.exact_proof_mask;
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
  decision.rebound_goal_probes = stats.rebound_goal_probes;
  decision.rebound_goal_hits = stats.rebound_goal_hits;
  decision.rebound_loss_hits = stats.rebound_loss_hits;
  decision.root_rebound_probes = stats.root_rebound_probes;
  decision.root_rebound_win_hits = stats.root_rebound_win_hits;
  decision.root_rebound_loss_hits = stats.root_rebound_loss_hits;
  decision.leaf_rebound_probes = stats.leaf_rebound_probes;
  decision.leaf_rebound_win_hits = stats.leaf_rebound_win_hits;
  decision.leaf_rebound_loss_hits = stats.leaf_rebound_loss_hits;
  decision.exchange_ply1_probes = stats.exchange_ply1_probes;
  decision.exchange_ply1_win_hits = stats.exchange_ply1_win_hits;
  decision.exchange_ply1_loss_hits = stats.exchange_ply1_loss_hits;
  decision.exchange_ply1_cutoffs = stats.exchange_ply1_cutoffs;
  decision.exchange_ply2_probes = stats.exchange_ply2_probes;
  decision.exchange_ply2_win_hits = stats.exchange_ply2_win_hits;
  decision.exchange_ply2_loss_hits = stats.exchange_ply2_loss_hits;
  decision.exchange_ply2_cutoffs = stats.exchange_ply2_cutoffs;
  decision.budget_exhausted = stats.budget_exhausted;
  return decision;
}

}  // namespace papersoccer::rank4_jacek_gate
