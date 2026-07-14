#include "papersoccer/bot.hpp"

#include <memory>
#include <stdexcept>

namespace papersoccer {

std::string_view bot_kind_name(BotKind kind) noexcept {
  switch (kind) {
    case BotKind::Random:
      return "RandomBot";
    case BotKind::Mcts:
      return "MctsBot";
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
  }
  throw std::invalid_argument("unknown bot kind");
}

}  // namespace papersoccer
