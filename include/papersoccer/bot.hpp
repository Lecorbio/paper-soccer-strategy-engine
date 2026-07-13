#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

#include "papersoccer/types.hpp"

namespace papersoccer {

enum class BotKind {
  Random,
  Mcts,
};

std::string_view bot_kind_name(BotKind kind) noexcept;

class Bot {
 public:
  virtual ~Bot() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual Move choose_move(const GameState &state) = 0;
};

class RandomBot final : public Bot {
 public:
  static constexpr std::uint64_t default_seed() noexcept { return 0xC0FFEE1234ULL; }

  explicit RandomBot(std::uint64_t seed = default_seed()) noexcept;

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;

 private:
  std::uint64_t state_;

  std::uint64_t next_random() noexcept;
};

struct BotConfig {
  BotKind kind{BotKind::Random};
  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t mcts_iterations{2000};
};

struct MctsConfig {
  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t iterations{2000};
  double exploration{1.4142135623730951};
};

struct SearchStats {
  std::uint32_t iterations{};
  std::size_t nodes{};
  std::uint64_t simulated_plies{};
  double root_value{};
};

class MctsSearch {
 public:
  MctsSearch(const GameState &root, std::uint64_t seed,
             double exploration = 1.4142135623730951);
  ~MctsSearch();

  MctsSearch(const MctsSearch &) = delete;
  MctsSearch &operator=(const MctsSearch &) = delete;

  void run_iterations(std::uint32_t count);
  Move best_move() const;
  SearchStats stats() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class MctsBot final : public Bot {
 public:
  explicit MctsBot(MctsConfig config = {});

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;
  SearchStats last_search_stats() const noexcept;

 private:
  MctsConfig config_;
  SearchStats last_search_stats_{};
};

std::unique_ptr<Bot> make_bot(const BotConfig &config);

}  // namespace papersoccer
