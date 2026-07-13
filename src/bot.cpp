#include "papersoccer/bot.hpp"

#include <stdexcept>
#include <vector>

#include "papersoccer/rules.hpp"

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

RandomBot::RandomBot(std::uint64_t seed) noexcept : state_(seed) {}

std::string_view RandomBot::name() const noexcept { return "RandomBot"; }

Move RandomBot::choose_move(const GameState &state) {
  const std::vector<Move> moves = legal_moves(state);
  if (moves.empty()) {
    throw std::invalid_argument("bot cannot choose move without legal moves");
  }

  const std::size_t index = static_cast<std::size_t>(next_random() % moves.size());
  return moves[index];
}

std::uint64_t RandomBot::next_random() noexcept {
  // SplitMix64 keeps the seeded sequence deterministic across platforms.
  state_ += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = state_;
  z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31U);
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
