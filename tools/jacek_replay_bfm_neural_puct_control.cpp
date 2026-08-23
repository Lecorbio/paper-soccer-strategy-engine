#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "../submissions/codingame/bots/neural_puct/bot.cpp"

#include "jacek_replay_bfm_gate_internal.hpp"

#include <chrono>
#include <limits>

namespace papersoccer::jacek_replay_gate {

ControlDecision choose_neural_puct_control(const GameState &state,
                                           const ControlConfig &config) {
  namespace engine = papersoccer::neural_puct;
  engine::SearchConfig search_config;
  search_config.max_nodes = config.tree_cap;
  search_config.max_simulations = config.work_cap == 0
                                      ? std::numeric_limits<std::uint64_t>::max()
                                      : config.work_cap;
  search_config.c_puct = engine::kDefaultCpuct;
  if (config.time_ms != 0) {
    search_config.absolute_deadline =
        engine::SearchClock::now() +
        std::chrono::milliseconds(config.time_ms);
  }
  engine::NeuralPuctSearch search(state, search_config);
  ControlDecision result;
  result.action = search.run();
  const engine::SearchStats &stats = search.stats();
  result.work = stats.simulations;
  result.tree_nodes = stats.nodes;
  result.neural_evaluations = stats.neural_evaluations;
  result.depth = stats.max_depth;
  result.root_value = stats.root_value;
  result.deadline_reached = stats.deadline_reached;
  result.cap_reached = stats.node_cap_reached;
  return result;
}

}  // namespace papersoccer::jacek_replay_gate
