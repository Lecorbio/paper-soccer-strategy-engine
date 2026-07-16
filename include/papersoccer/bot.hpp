#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer {

enum class BotKind {
  Random,
  Mcts,
  AlphaBeta,
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

struct AlphaBetaConfig {
  static constexpr std::uint32_t maximum_turn_depth{64};
  static constexpr std::uint32_t maximum_search_plies{512};

  std::uint32_t max_turn_depth{6};
  std::uint64_t max_nodes{100'000};
  std::size_t transposition_table_entries{65'536};
  std::uint32_t max_search_plies{10};
};

enum class AlphaBetaScoreBound {
  Exact,
  Lower,
  Upper,
};

struct AlphaBetaRootMove {
  Move move{};
  int score{};
  AlphaBetaScoreBound bound{AlphaBetaScoreBound::Exact};
};

struct AlphaBetaSearchStats {
  std::uint32_t completed_turn_depth{};
  std::uint32_t attempted_turn_depth{};
  std::uint64_t nodes{};
  std::uint64_t leaf_evaluations{};
  std::uint64_t terminal_nodes{};
  std::uint64_t cutoffs{};
  std::uint64_t transposition_probes{};
  std::uint64_t transposition_hits{};
  std::uint64_t transposition_cutoffs{};
  std::uint64_t transposition_stores{};
  std::uint64_t physical_ply_cutoffs{};
  std::uint32_t max_physical_ply{};
  int root_score{};
  bool budget_exhausted{};
  std::vector<Move> principal_variation{};
  std::vector<AlphaBetaRootMove> root_moves{};
};

class AlphaBetaBot final : public Bot {
 public:
  explicit AlphaBetaBot(AlphaBetaConfig config = {});

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;
  const AlphaBetaConfig &config() const noexcept;
  const AlphaBetaSearchStats &last_search_stats() const noexcept;

 private:
  AlphaBetaConfig config_;
  AlphaBetaSearchStats last_search_stats_{};
};

struct BotConfig {
  BotKind kind{BotKind::Random};
  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t mcts_iterations{2000};
  std::uint32_t alpha_beta_depth{6};
  std::uint64_t alpha_beta_max_nodes{100'000};
  std::size_t alpha_beta_transposition_table_entries{65'536};
  std::uint32_t alpha_beta_max_search_plies{10};
};

enum class MctsRolloutPolicy {
  Uniform,
  Tactical,
};

enum class MctsLeafPolicy {
  RolloutOnly,
  TacticalQuiescence,
};

struct MctsConfig {
  static constexpr std::uint32_t default_quiescence_max_depth{8};
  static constexpr std::uint32_t default_quiescence_max_nodes{256};
  static constexpr std::uint32_t maximum_quiescence_max_depth{64};
  static constexpr std::uint32_t maximum_quiescence_max_nodes{1'000'000};

  std::uint64_t seed{RandomBot::default_seed()};
  std::uint32_t iterations{2000};
  double exploration{1.4142135623730951};
  MctsRolloutPolicy rollout_policy{MctsRolloutPolicy::Tactical};
  bool reuse_tree{true};
  std::size_t max_nodes{65536};
  MctsLeafPolicy leaf_policy{MctsLeafPolicy::RolloutOnly};
  std::uint32_t quiescence_max_depth{default_quiescence_max_depth};
  std::uint32_t quiescence_max_nodes{default_quiescence_max_nodes};
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
  std::uint64_t tactical_probes{};
  std::uint64_t tactical_nodes{};
  std::uint64_t tactical_solved_positions{};
  std::uint64_t tactical_depth_cutoffs{};
  std::uint64_t tactical_node_cutoffs{};
  std::uint32_t max_tactical_depth{};
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
