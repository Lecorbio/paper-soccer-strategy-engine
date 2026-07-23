#pragma once

#include "papersoccer/bot.hpp"

namespace papersoccer::detail {

enum class AlphaBetaLeafEvaluator {
  Handcrafted,
  JacekNeural,
};

void validate_alpha_beta_config(const AlphaBetaConfig &config);

Move run_alpha_beta_search(const GameState &state,
                           const AlphaBetaConfig &config,
                           AlphaBetaLeafEvaluator evaluator,
                           AlphaBetaSearchStats &stats);

}  // namespace papersoccer::detail
