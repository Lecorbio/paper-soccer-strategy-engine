#include "papersoccer/bot.hpp"

#include <memory>
#include <stdexcept>

#include "papersoccer/game_review.hpp"

namespace papersoccer {

std::string_view bot_kind_name(BotKind kind) noexcept {
  switch (kind) {
    case BotKind::Random:
      return "RandomBot";
    case BotKind::Mcts:
      return "MctsBot";
    case BotKind::AlphaBeta:
      return "AlphaBetaBot";
    case BotKind::JacekInspired:
      return "JacekInspiredBot";
    case BotKind::Rank5Derived:
      return "Rank5DerivedBot";
    case BotKind::DeepTurnSearch:
      return "DeepTurnSearchBot";
    case BotKind::JacekReplayBfm:
      return "JacekReplayBfmBot";
  }
  return "UnknownBot";
}

std::unique_ptr<Bot> make_bot(const BotConfig &config) {
  switch (config.kind) {
    case BotKind::Random:
      return std::make_unique<RandomBot>(config.seed);
    case BotKind::Mcts: {
      MctsConfig mcts_config;
      mcts_config.seed = config.seed;
      mcts_config.iterations = config.mcts_iterations;
      return std::make_unique<MctsBot>(mcts_config);
    }
    case BotKind::AlphaBeta: {
      AlphaBetaConfig alpha_beta_config;
      alpha_beta_config.max_turn_depth = config.alpha_beta_depth;
      alpha_beta_config.max_nodes = config.alpha_beta_max_nodes;
      alpha_beta_config.transposition_table_entries =
          config.alpha_beta_transposition_table_entries;
      alpha_beta_config.max_search_plies =
          config.alpha_beta_max_search_plies;
      alpha_beta_config.max_time_ms = config.alpha_beta_max_time_ms;
      return std::make_unique<AlphaBetaBot>(alpha_beta_config);
    }
    case BotKind::JacekInspired: {
      AlphaBetaConfig search_config;
      search_config.max_turn_depth = config.alpha_beta_depth;
      search_config.max_nodes = config.alpha_beta_max_nodes;
      search_config.transposition_table_entries =
          config.alpha_beta_transposition_table_entries;
      search_config.max_search_plies =
          config.alpha_beta_max_search_plies;
      search_config.max_time_ms = config.alpha_beta_max_time_ms;
      return std::make_unique<JacekInspiredBot>(search_config);
    }
    case BotKind::Rank5Derived:
      return std::make_unique<Rank5DerivedBot>();
    case BotKind::DeepTurnSearch:
      return std::make_unique<DeepTurnSearchBot>(
          GameReviewConfig::locked(ReviewMode::Deep).deep_profile);
    case BotKind::JacekReplayBfm:
      return std::make_unique<JacekReplayBfmBot>(config.jacek_replay_bfm);
  }
  throw std::invalid_argument("unknown bot kind");
}

}  // namespace papersoccer
