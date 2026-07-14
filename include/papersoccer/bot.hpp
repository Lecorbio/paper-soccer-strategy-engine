#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
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

enum class MctsRolloutPolicy {
  Uniform,
  Tactical,
};

struct MctsConfig {
  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t iterations{2000};
  double exploration{1.4142135623730951};
  MctsRolloutPolicy rollout_policy{MctsRolloutPolicy::Tactical};
  bool reuse_tree{true};
  std::size_t max_nodes{65536};
};

struct SearchStats {
  std::uint32_t iterations{};
  std::size_t nodes{};
  std::uint64_t simulated_plies{};
  double root_value{};
  std::uint64_t total_root_visits{};
  std::uint64_t reused_visits{};
  std::uint32_t max_depth{};
  std::size_t proven_nodes{};
  std::optional<Player> proven_winner{};
  std::uint64_t rebuild_count{};
  bool expansion_saturated{};
};

class MctsSearch {
 public:
  MctsSearch(const GameState &root, std::uint64_t seed,
             double exploration = 1.4142135623730951);
  MctsSearch(const GameState &root, const MctsConfig &config);
  ~MctsSearch();

  MctsSearch(const MctsSearch &) = delete;
  MctsSearch &operator=(const MctsSearch &) = delete;

  void run_iterations(std::uint32_t count);
  Move best_move() const;
  SearchStats stats() const noexcept;

 private:
  class Impl;
  friend class MctsBot;

  explicit MctsSearch(std::unique_ptr<Impl> impl);
  std::unique_ptr<MctsSearch> clone() const;
  bool rebase(const GameState &root);
  void set_rebuild_count(std::uint64_t count) noexcept;

  std::unique_ptr<Impl> impl_;
};

class MctsBot final : public Bot {
 public:
  explicit MctsBot(MctsConfig config = {});
  MctsBot(const MctsBot &other);
  MctsBot &operator=(const MctsBot &other);
  MctsBot(MctsBot &&) noexcept = default;
  MctsBot &operator=(MctsBot &&) noexcept = default;

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;
  SearchStats last_search_stats() const noexcept;

 private:
  MctsConfig config_;
  SearchStats last_search_stats_{};
  std::unique_ptr<MctsSearch> search_{};
  std::uint64_t rebuild_count_{};
};

std::unique_ptr<Bot> make_bot(const BotConfig &config);

}  // namespace papersoccer
