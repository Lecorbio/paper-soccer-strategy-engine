#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "papersoccer/arena.hpp"

namespace papersoccer::arena::detail {

constexpr std::uint64_t kBootstrapSeedSalt = 0x4152454e414349ULL;
constexpr std::uint64_t kOpeningSeedSalt = 0x4f50454e494e4753ULL;
constexpr std::uint64_t kWarmupSeedSalt = 0x5741524d555053ULL;
constexpr std::size_t kMaxPositionGenerationAttempts = 4096;
constexpr std::size_t kWarmupGenerationPlies = 24;

constexpr std::string_view runtime_name() noexcept {
#ifdef __EMSCRIPTEN__
  return "wasm";
#else
  return "native";
#endif
}

enum class Entrant { Candidate, Reference };

struct SplitMix64 {
  std::uint64_t state{};

  std::uint64_t next() noexcept {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  std::size_t index(std::size_t upper_bound) noexcept {
    const auto bound = static_cast<std::uint64_t>(upper_bound);
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    std::uint64_t value = 0;
    do {
      value = next();
    } while (value < threshold);
    return static_cast<std::size_t>(value % bound);
  }
};

struct Participant {
  Entrant entrant{Entrant::Candidate};
  Player color{Player::One};
  std::uint64_t seed{};
  ArenaBotConfig config{};
};

struct DecisionReport {
  std::size_t ply{};
  Entrant entrant{Entrant::Candidate};
  Player player{Player::One};
  Point from{};
  Move move{};
  std::uint64_t elapsed_ns{};
  std::optional<SearchStats> stats{};
  std::optional<AlphaBetaSearchStats> alpha_beta_stats{};
  std::optional<Rank5DerivedSearchStats> rank5_derived_stats{};
};

struct GameReport {
  std::size_t pair_index{};
  std::size_t game_in_pair{};
  Participant player_one{};
  Participant player_two{};
  Status status{Status::InProgress};
  std::optional<Entrant> winning_entrant{};
  std::size_t plies{};
  bool truncated{};
  std::vector<DecisionReport> decisions{};
};

struct OpeningReport {
  std::size_t pair_index{};
  std::uint64_t generation_seed{};
  std::size_t attempts{};
  std::string opening_id{};
  std::string phase{};
  std::string state_hash{};
  std::string canonical_key{};
  std::vector<Move> transcript{};
  GameState state{};
};

struct PositionEvaluation {
  Entrant entrant{Entrant::Candidate};
  std::uint64_t seed{};
  Move move{};
  std::uint64_t elapsed_ns{};
  std::optional<SearchStats> stats{};
  std::optional<AlphaBetaSearchStats> alpha_beta_stats{};
  std::optional<Rank5DerivedSearchStats> rank5_derived_stats{};
};

struct PositionReport {
  std::size_t index{};
  std::uint64_t generation_seed{};
  GameState state{};
  PositionEvaluation candidate{};
  PositionEvaluation reference{};
};

struct Record {
  std::size_t games{};
  std::size_t wins{};
  std::size_t losses{};
  std::size_t truncations{};

  std::size_t scored_games() const noexcept { return wins + losses; }
  double score() const noexcept { return static_cast<double>(wins); }
};

struct TimingSummary {
  std::size_t decisions{};
  std::uint64_t total_ns{};
  std::uint64_t min_ns{};
  std::uint64_t median_ns{};
  std::uint64_t p90_ns{};
  std::uint64_t p95_ns{};
  std::uint64_t p99_ns{};
  std::uint64_t max_ns{};
  double median_iterations_per_second{};
  double median_simulated_plies_per_second{};
  double median_nodes_per_second{};
};

struct Rank5DerivedCounterSummary {
  std::uint64_t sum{};
  std::uint64_t max{};
};

struct Rank5DerivedSummary {
  std::size_t decisions{};
  std::size_t fresh_root_searches{};
  std::size_t cached_continuation_edges{};
  std::uint64_t visited_nodes_sum{};
  std::uint64_t requested_nodes_sum{};
  std::size_t budget_exhausted_fresh_searches{};
  std::map<std::uint32_t, std::size_t> completed_depth_histogram{};
  std::map<std::uint32_t, std::size_t> attempted_depth_histogram{};
  std::map<std::size_t, std::size_t> planned_action_length_histogram{};
  std::size_t maximum_current_edge_index{};
  std::optional<int> minimum_root_score{};
  std::optional<int> maximum_root_score{};
  Rank5DerivedCounterSummary leaf_evaluations{};
  Rank5DerivedCounterSummary terminal_nodes{};
  Rank5DerivedCounterSummary completed_actions{};
  Rank5DerivedCounterSummary cutoffs{};
  Rank5DerivedCounterSummary transposition_probes{};
  Rank5DerivedCounterSummary transposition_hits{};
  Rank5DerivedCounterSummary transposition_cutoffs{};
  Rank5DerivedCounterSummary transposition_stores{};
  Rank5DerivedCounterSummary continuation_transposition_hits{};
  Rank5DerivedCounterSummary evaluation_cache_probes{};
  Rank5DerivedCounterSummary evaluation_cache_hits{};
  Rank5DerivedCounterSummary terminal_bound_cutoffs{};
  Rank5DerivedCounterSummary forced_edges{};
  Rank5DerivedCounterSummary root_seed_actions{};
  Rank5DerivedCounterSummary root_transposition_reuses{};
  Rank5DerivedCounterSummary max_action_edges{};
  TimingSummary fresh_root_timing{};
  TimingSummary all_edge_timing{};
};

struct MctsSummary {
  std::size_t searches{};
  std::uint64_t iterations{};
  std::uint64_t nodes_sum{};
  std::uint64_t simulated_plies{};
  std::uint64_t total_root_visits_sum{};
  std::uint64_t reused_visits_sum{};
  std::uint32_t max_depth{};
  std::uint64_t proven_nodes_sum{};
  std::size_t proven_searches{};
  std::uint64_t tactical_probes{};
  std::uint64_t tactical_nodes{};
  std::uint64_t tactical_solved_positions{};
  std::uint64_t tactical_depth_cutoffs{};
  std::uint64_t tactical_node_cutoffs{};
  std::uint32_t max_tactical_depth{};
  std::uint64_t rebuild_count_max{};
  std::size_t expansion_saturated_searches{};
};

struct AlphaBetaSummary {
  static constexpr std::size_t depth_bucket_count =
      static_cast<std::size_t>(AlphaBetaConfig::maximum_turn_depth) + 1U;

  std::size_t searches{};
  std::uint64_t nodes_sum{};
  std::uint64_t leaf_evaluations_sum{};
  std::uint64_t terminal_nodes_sum{};
  std::uint64_t cutoffs_sum{};
  std::uint64_t transposition_probes_sum{};
  std::uint64_t transposition_hits_sum{};
  std::uint64_t transposition_cutoffs_sum{};
  std::uint64_t transposition_stores_sum{};
  std::uint64_t physical_ply_cutoffs_sum{};
  std::uint32_t max_completed_turn_depth{};
  std::uint32_t max_attempted_turn_depth{};
  std::uint32_t max_physical_ply{};
  std::size_t budget_exhausted_searches{};
  std::array<std::size_t, depth_bucket_count>
      completed_turn_depth_histogram{};
  std::array<std::size_t, depth_bucket_count>
      attempted_turn_depth_histogram{};
};

struct BootstrapInterval {
  std::uint64_t seed{};
  std::size_t samples{};
  std::size_t valid_pairs{};
  std::size_t invalid_pairs{};
  std::optional<double> lower_percent{};
  std::optional<double> upper_percent{};
};

std::string_view entrant_name(Entrant entrant) noexcept;
std::string_view player_name(Player player) noexcept;
std::string_view status_name(Status status) noexcept;
std::string_view policy_name(MctsRolloutPolicy policy) noexcept;
std::string_view leaf_policy_name(MctsLeafPolicy policy) noexcept;
std::string_view kind_name(BotKind kind) noexcept;
std::string_view alpha_beta_bound_name(AlphaBetaScoreBound bound) noexcept;

void validate_common(const RulesConfig &rules,
                     const ArenaBotConfig &candidate,
                     const ArenaBotConfig &reference);
GameReport play_game(std::size_t pair_index, std::size_t game_in_pair,
                     Participant player_one, Participant player_two,
                     const GameState &initial_state, std::size_t max_plies);
OpeningReport generate_opening(std::size_t pair_index,
                               const RulesConfig &rules,
                               std::uint64_t pair_seed,
                               std::size_t opening_plies);
std::vector<OpeningReport> validate_frozen_openings(
    const RulesConfig &rules,
    const std::vector<FrozenOpening> &frozen_openings);
GameState generate_position(const RulesConfig &rules, std::uint64_t seed,
                            std::size_t generation_plies);
void warm_up_match_entrants(const RulesConfig &rules,
                            const ArenaBotConfig &candidate,
                            const ArenaBotConfig &reference,
                            std::uint64_t base_seed,
                            std::size_t decisions_per_entrant);
PositionEvaluation evaluate_position(const ArenaBotConfig &config,
                                     Entrant entrant, std::uint64_t seed,
                                     const GameState &state);

TimingSummary summarize_timing(
    const std::vector<const DecisionReport *> &decisions);
MctsSummary summarize_mcts(
    const std::vector<const DecisionReport *> &decisions);
AlphaBetaSummary summarize_alpha_beta(
    const std::vector<const DecisionReport *> &decisions);
Rank5DerivedSummary summarize_rank5_derived(
    const std::vector<const DecisionReport *> &decisions);
Record record_for(const std::vector<GameReport> &games, Entrant entrant,
                  std::optional<Player> color = std::nullopt);
std::vector<const DecisionReport *> decisions_for(
    const std::vector<GameReport> &games, Entrant entrant);
std::vector<const DecisionReport *> decisions_for(
    const std::vector<PositionReport> &positions, Entrant entrant,
    std::vector<DecisionReport> &storage);
std::vector<std::optional<double>> candidate_pair_scores(
    const std::vector<GameReport> &games, std::size_t pair_count);
BootstrapInterval bootstrap_interval(
    const std::vector<std::optional<double>> &pair_scores,
    std::uint64_t base_seed,
    std::size_t samples);

}  // namespace papersoccer::arena::detail
