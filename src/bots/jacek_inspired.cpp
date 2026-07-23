#include "papersoccer/bot.hpp"

#include "alpha_beta_internal.hpp"
#include "jacek_neural_model.hpp"

namespace papersoccer {

JacekInspiredBot::JacekInspiredBot(AlphaBetaConfig config)
    : config_(config) {
  detail::validate_alpha_beta_config(config_);
}

std::string_view JacekInspiredBot::name() const noexcept {
  return "JacekInspiredBot";
}

Move JacekInspiredBot::choose_move(const GameState &state) {
  return detail::run_alpha_beta_search(
      state, config_, detail::AlphaBetaLeafEvaluator::JacekNeural,
      last_search_stats_);
}

const AlphaBetaConfig &JacekInspiredBot::config() const noexcept {
  return config_;
}

const AlphaBetaSearchStats &JacekInspiredBot::last_search_stats() const
    noexcept {
  return last_search_stats_;
}

std::string_view JacekInspiredBot::model_sha256() noexcept {
  return detail::jacek_neural_model::kModelSha256;
}

}  // namespace papersoccer
