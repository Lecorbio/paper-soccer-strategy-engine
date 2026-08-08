#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "papersoccer/rules.hpp"

namespace papersoccer::arena::detail {

using Clock = std::chrono::steady_clock;

namespace {

bool same_rules(const RulesConfig &left, const RulesConfig &right) noexcept {
  return left.width == right.width && left.height == right.height &&
         left.goal_rule == right.goal_rule &&
         left.blocked_rule == right.blocked_rule;
}

bool same_state(const GameState &left, const GameState &right) {
  return same_rules(left.config, right.config) && left.ball == right.ball &&
         left.to_move == right.to_move && left.status == right.status &&
         left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

std::string frozen_context(std::size_t index) {
  return "arena frozen opening " + std::to_string(index);
}

void add_rank5_counter(Rank5DerivedCounterSummary &summary,
                       std::uint64_t value) noexcept {
  summary.sum += value;
  summary.max = std::max(summary.max, value);
}

}  // namespace

std::string_view entrant_name(Entrant entrant) noexcept {
  return entrant == Entrant::Candidate ? "candidate" : "reference";
}

std::string_view player_name(Player player) noexcept {
  return player == Player::One ? "one" : "two";
}

std::string_view status_name(Status status) noexcept {
  switch (status) {
    case Status::InProgress:
      return "in_progress";
    case Status::WonByOne:
      return "won_by_one";
    case Status::WonByTwo:
      return "won_by_two";
  }
  return "unknown";
}

std::string_view policy_name(MctsRolloutPolicy policy) noexcept {
  return policy == MctsRolloutPolicy::Uniform ? "uniform" : "tactical";
}

std::string_view leaf_policy_name(MctsLeafPolicy policy) noexcept {
  return policy == MctsLeafPolicy::RolloutOnly ? "rollout_only"
                                               : "tactical_quiescence";
}

std::string_view kind_name(BotKind kind) noexcept {
  switch (kind) {
    case BotKind::Random:
      return "random";
    case BotKind::Mcts:
      return "mcts";
    case BotKind::AlphaBeta:
      return "alpha-beta";
    case BotKind::JacekInspired:
      return "jacek-inspired";
    case BotKind::Rank5Derived:
      return "rank5-derived";
    case BotKind::DeepTurnSearch:
      return "deep-turn-search";
  }
  return "unknown";
}

std::string_view alpha_beta_bound_name(AlphaBetaScoreBound bound) noexcept {
  switch (bound) {
    case AlphaBetaScoreBound::Exact:
      return "exact";
    case AlphaBetaScoreBound::Lower:
      return "lower";
    case AlphaBetaScoreBound::Upper:
      return "upper";
  }
  return "unknown";
}

void validate_rules(const RulesConfig &rules) {
  if (rules.width < 2 || rules.height < 2) {
    throw std::invalid_argument("arena board width and height must be at least 2");
  }
}

void validate_bot_config(const ArenaBotConfig &config) {
  switch (config.kind) {
    case BotKind::Random:
      return;
    case BotKind::Mcts:
      if (config.leaf_policy != MctsLeafPolicy::RolloutOnly &&
          config.leaf_policy != MctsLeafPolicy::TacticalQuiescence) {
        throw std::invalid_argument("arena MCTS leaf policy is unknown");
      }
      if (config.iterations == 0) {
        throw std::invalid_argument(
            "arena MCTS iterations must be greater than zero");
      }
      if (!std::isfinite(config.exploration) || config.exploration < 0.0) {
        throw std::invalid_argument(
            "arena MCTS exploration must be finite and non-negative");
      }
      if (config.max_nodes < 2) {
        throw std::invalid_argument("arena MCTS max nodes must be at least 2");
      }
      if (config.quiescence_max_depth == 0 ||
          config.quiescence_max_depth >
              MctsConfig::maximum_quiescence_max_depth) {
        throw std::invalid_argument(
            "arena MCTS quiescence max depth must be between 1 and " +
            std::to_string(MctsConfig::maximum_quiescence_max_depth));
      }
      if (config.quiescence_max_nodes == 0 ||
          config.quiescence_max_nodes >
              MctsConfig::maximum_quiescence_max_nodes) {
        throw std::invalid_argument(
            "arena MCTS quiescence max nodes must be between 1 and " +
            std::to_string(MctsConfig::maximum_quiescence_max_nodes));
      }
      if (config.leaf_policy == MctsLeafPolicy::TacticalQuiescence &&
          config.rollout_policy != MctsRolloutPolicy::Tactical) {
        throw std::invalid_argument(
            "arena tactical quiescence requires the tactical rollout policy");
      }
      return;
    case BotKind::AlphaBeta:
    case BotKind::JacekInspired:
      if (config.alpha_beta_depth == 0 ||
          config.alpha_beta_depth > AlphaBetaConfig::maximum_turn_depth) {
        throw std::invalid_argument(
            "arena alpha-beta depth must be between 1 and " +
            std::to_string(AlphaBetaConfig::maximum_turn_depth));
      }
      if (config.alpha_beta_max_nodes == 0) {
        throw std::invalid_argument(
            "arena alpha-beta max nodes must be greater than zero");
      }
      if (config.alpha_beta_max_search_plies == 0 ||
          config.alpha_beta_max_search_plies >
              AlphaBetaConfig::maximum_search_plies) {
        throw std::invalid_argument(
            "arena alpha-beta max search plies must be between 1 and " +
            std::to_string(AlphaBetaConfig::maximum_search_plies));
      }
      return;
    case BotKind::Rank5Derived:
      return;
    case BotKind::DeepTurnSearch:
      (void)CompleteTurnAnalysisConfig::deep(
          config.complete_turn_max_nodes);
      return;
  }
  throw std::invalid_argument("arena bot kind is unknown");
}

void validate_common(const RulesConfig &rules, const ArenaBotConfig &candidate,
                     const ArenaBotConfig &reference) {
  validate_rules(rules);
  validate_bot_config(candidate);
  validate_bot_config(reference);
}

std::unique_ptr<Bot> make_arena_bot(const ArenaBotConfig &config,
                                    std::uint64_t seed) {
  switch (config.kind) {
    case BotKind::Random:
      return std::make_unique<RandomBot>(seed);
    case BotKind::Mcts: {
      MctsConfig mcts;
      mcts.seed = seed;
      mcts.iterations = config.iterations;
      mcts.exploration = config.exploration;
      mcts.rollout_policy = config.rollout_policy;
      mcts.reuse_tree = config.reuse_tree;
      mcts.max_nodes = config.max_nodes;
      mcts.leaf_policy = config.leaf_policy;
      mcts.quiescence_max_depth = config.quiescence_max_depth;
      mcts.quiescence_max_nodes = config.quiescence_max_nodes;
      return std::make_unique<MctsBot>(mcts);
    }
    case BotKind::AlphaBeta:
    case BotKind::JacekInspired: {
      AlphaBetaConfig alpha_beta;
      alpha_beta.max_turn_depth = config.alpha_beta_depth;
      alpha_beta.max_nodes = config.alpha_beta_max_nodes;
      alpha_beta.transposition_table_entries =
          config.alpha_beta_transposition_table_entries;
      alpha_beta.max_search_plies = config.alpha_beta_max_search_plies;
      if (config.kind == BotKind::JacekInspired) {
        return std::make_unique<JacekInspiredBot>(alpha_beta);
      }
      return std::make_unique<AlphaBetaBot>(alpha_beta);
    }
    case BotKind::Rank5Derived:
      return std::make_unique<Rank5DerivedBot>();
    case BotKind::DeepTurnSearch:
      return std::make_unique<DeepTurnSearchBot>(
          config.complete_turn_max_nodes);
  }
  throw std::invalid_argument("arena bot kind is unknown");
}

std::optional<SearchStats> search_stats(Bot &bot) {
  auto *mcts = dynamic_cast<MctsBot *>(&bot);
  if (mcts == nullptr) {
    return std::nullopt;
  }
  return mcts->last_search_stats();
}

std::optional<AlphaBetaSearchStats> alpha_beta_search_stats(Bot &bot) {
  if (auto *alpha_beta = dynamic_cast<AlphaBetaBot *>(&bot)) {
    return alpha_beta->last_search_stats();
  }
  if (auto *jacek = dynamic_cast<JacekInspiredBot *>(&bot)) {
    return jacek->last_search_stats();
  }
  return std::nullopt;
}

std::optional<Rank5DerivedSearchStats> rank5_derived_search_stats(Bot &bot) {
  if (auto *rank5_derived = dynamic_cast<Rank5DerivedBot *>(&bot)) {
    return rank5_derived->last_search_stats();
  }
  return std::nullopt;
}

std::optional<CompleteTurnSearchStats> deep_turn_search_stats(Bot &bot) {
  if (auto *deep = dynamic_cast<DeepTurnSearchBot *>(&bot)) {
    return deep->last_search_stats();
  }
  return std::nullopt;
}

std::optional<std::uint64_t> deep_turn_search_profile_nodes(Bot &bot) {
  if (auto *deep = dynamic_cast<DeepTurnSearchBot *>(&bot)) {
    return deep->config().max_nodes;
  }
  return std::nullopt;
}

bool contains_move(const std::vector<Move> &moves, Move move) {
  return std::find(moves.begin(), moves.end(), move) != moves.end();
}

std::uint64_t elapsed_nanoseconds(Clock::time_point start,
                                  Clock::time_point end) noexcept {
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  return elapsed <= 0 ? 0 : static_cast<std::uint64_t>(elapsed);
}

DecisionReport choose_and_measure(Bot &bot, Entrant entrant, std::size_t ply,
                                  const GameState &state) {
  const Point from = state.ball;
  const auto start = Clock::now();
  const Move move = bot.choose_move(state);
  const auto end = Clock::now();
  const std::vector<Move> moves = legal_moves(state);
  if (!contains_move(moves, move)) {
    throw std::logic_error(std::string(entrant_name(entrant)) +
                           " bot returned an illegal move at ply " +
                           std::to_string(ply));
  }
  return DecisionReport{ply,
                        entrant,
                        state.to_move,
                        from,
                        move,
                        elapsed_nanoseconds(start, end),
                        search_stats(bot),
                        alpha_beta_search_stats(bot),
                        rank5_derived_search_stats(bot),
                        deep_turn_search_stats(bot),
                        deep_turn_search_profile_nodes(bot)};
}

GameReport play_game(std::size_t pair_index, std::size_t game_in_pair,
                     Participant player_one, Participant player_two,
                     const GameState &initial_state, std::size_t max_plies) {
  if (initial_state.path.empty()) {
    throw std::invalid_argument("arena initial state path must not be empty");
  }
  if (is_terminal(initial_state)) {
    throw std::invalid_argument("arena initial state must be non-terminal");
  }
  const std::size_t initial_plies = initial_state.path.size() - 1;
  if (initial_plies >= max_plies) {
    throw std::invalid_argument(
        "arena initial state must precede the max ply limit");
  }

  GameReport report;
  report.pair_index = pair_index;
  report.game_in_pair = game_in_pair;
  report.player_one = player_one;
  report.player_two = player_two;
  report.decisions.reserve(
      std::min<std::size_t>(max_plies - initial_plies, 512));

  std::unique_ptr<Bot> player_one_bot =
      make_arena_bot(player_one.config, player_one.seed);
  std::unique_ptr<Bot> player_two_bot =
      make_arena_bot(player_two.config, player_two.seed);
  GameState state = initial_state;

  while (!is_terminal(state) &&
         initial_plies + report.decisions.size() < max_plies) {
    const bool one_to_move = state.to_move == Player::One;
    Bot &bot = one_to_move ? *player_one_bot : *player_two_bot;
    const Entrant entrant = one_to_move ? player_one.entrant : player_two.entrant;
    DecisionReport decision = choose_and_measure(
        bot, entrant, initial_plies + report.decisions.size() + 1, state);
    state = apply_move(state, decision.move);
    report.decisions.push_back(std::move(decision));
  }

  report.status = state.status;
  report.plies = initial_plies + report.decisions.size();
  report.truncated = !is_terminal(state);
  if (!report.truncated) {
    const std::optional<Player> winning_player = winner(state);
    if (!winning_player.has_value()) {
      throw std::logic_error("terminal arena game has no winner");
    }
    report.winning_entrant =
        *winning_player == Player::One ? player_one.entrant : player_two.entrant;
  }
  return report;
}

OpeningReport generate_opening(std::size_t pair_index,
                               const RulesConfig &rules,
                               std::uint64_t pair_seed,
                               std::size_t opening_plies) {
  SplitMix64 attempt_seeds{pair_seed};
  for (std::size_t attempt = 0; attempt < kMaxPositionGenerationAttempts;
       ++attempt) {
    const std::uint64_t generation_seed = attempt_seeds.next();
    GameState state = generate_position(rules, generation_seed, opening_plies);
    const std::size_t actual_plies = state.path.empty() ? 0 : state.path.size() - 1;
    if (!is_terminal(state) && actual_plies == opening_plies) {
      OpeningReport report;
      report.pair_index = pair_index;
      report.generation_seed = generation_seed;
      report.attempts = attempt + 1;
      report.transcript.reserve(actual_plies);
      for (std::size_t index = 1; index < state.path.size(); ++index) {
        report.transcript.push_back(Move{state.path[index]});
      }
      report.state = std::move(state);
      return report;
    }
  }
  throw std::runtime_error(
      "could not generate a non-terminal arena opening at the requested ply");
}

std::vector<OpeningReport> validate_frozen_openings(
    const RulesConfig &rules,
    const std::vector<FrozenOpening> &frozen_openings) {
  std::vector<OpeningReport> reports;
  if (frozen_openings.empty()) {
    return reports;
  }

  const std::size_t expected_depth = frozen_openings.front().depth;
  const std::string &expected_phase = frozen_openings.front().phase;
  if (expected_depth == 0 || expected_phase.empty()) {
    throw std::invalid_argument(
        "arena frozen openings require a nonzero depth and phase");
  }
  std::unordered_set<std::string> opening_ids;
  std::unordered_set<std::string> state_hashes;
  std::unordered_set<std::string> canonical_keys;
  reports.reserve(frozen_openings.size());
  for (std::size_t index = 0; index < frozen_openings.size(); ++index) {
    const FrozenOpening &opening = frozen_openings[index];
    const std::string context = frozen_context(index);
    if (opening.opening_id.empty() || opening.state_hash.empty() ||
        opening.canonical_key.empty()) {
      throw std::invalid_argument(context + " is missing identity metadata");
    }
    if (opening.phase != expected_phase || opening.depth != expected_depth ||
        opening.transcript.size() != expected_depth) {
      throw std::invalid_argument(context +
                                  " has mismatched phase, depth, or transcript");
    }
    if (!opening_ids.insert(opening.opening_id).second ||
        !state_hashes.insert(opening.state_hash).second ||
        !canonical_keys.insert(opening.canonical_key).second) {
      throw std::invalid_argument(context + " duplicates frozen-bank metadata");
    }
    if (!same_rules(rules, opening.state.config)) {
      throw std::invalid_argument(context +
                                  " uses rules that differ from the arena");
    }

    GameState replayed = make_initial_state(rules);
    for (std::size_t ply = 0; ply < opening.transcript.size(); ++ply) {
      if (is_terminal(replayed)) {
        throw std::invalid_argument(context +
                                    " continues after a terminal position");
      }
      const std::vector<Move> moves = legal_moves(replayed);
      if (!contains_move(moves, opening.transcript[ply])) {
        throw std::invalid_argument(context + " has an illegal transcript edge at ply " +
                                    std::to_string(ply + 1));
      }
      replayed = apply_move(replayed, opening.transcript[ply]);
    }
    if (is_terminal(replayed) || legal_moves(replayed).empty()) {
      throw std::invalid_argument(context + " ends in a terminal position");
    }
    if (!same_state(replayed, opening.state)) {
      throw std::invalid_argument(context +
                                  " state does not match its exact transcript");
    }

    OpeningReport report;
    report.pair_index = index;
    report.generation_seed = opening.generation_seed;
    report.opening_id = opening.opening_id;
    report.phase = opening.phase;
    report.state_hash = opening.state_hash;
    report.canonical_key = opening.canonical_key;
    report.transcript = opening.transcript;
    report.state = opening.state;
    reports.push_back(std::move(report));
  }
  return reports;
}

GameState generate_position(const RulesConfig &rules, std::uint64_t seed,
                            std::size_t generation_plies) {
  GameState state = make_initial_state(rules);
  RandomBot generator(seed);
  for (std::size_t ply = 0; ply < generation_plies; ++ply) {
    if (is_terminal(state)) {
      return state;
    }
    state = apply_move(state, generator.choose_move(state));
  }
  return state;
}

namespace {

GameState generate_uniform_warmup_position(const RulesConfig &rules,
                                           std::uint64_t seed) {
  GameState state = make_initial_state(rules);
  SplitMix64 generator{seed};
  for (std::size_t ply = 0; ply < kWarmupGenerationPlies; ++ply) {
    if (is_terminal(state)) {
      return state;
    }
    const std::vector<Move> moves = legal_moves(state);
    if (moves.empty()) {
      return state;
    }
    state = apply_move(state, moves[generator.index(moves.size())]);
  }
  return state;
}

}  // namespace

void warm_up_match_entrants(const RulesConfig &rules,
                            const ArenaBotConfig &candidate_config,
                            const ArenaBotConfig &reference_config,
                            std::uint64_t base_seed,
                            std::size_t decisions_per_entrant) {
  if (decisions_per_entrant == 0) {
    return;
  }

  SplitMix64 seeds{base_seed ^ kWarmupSeedSalt};
  std::unique_ptr<Bot> candidate =
      make_arena_bot(candidate_config, seeds.next());
  std::unique_ptr<Bot> reference =
      make_arena_bot(reference_config, seeds.next());
  for (std::size_t index = 0; index < decisions_per_entrant; ++index) {
    SplitMix64 attempt_seeds{seeds.next()};
    std::optional<GameState> generated;
    for (std::size_t attempt = 0; attempt < kMaxPositionGenerationAttempts;
         ++attempt) {
      GameState state =
          generate_uniform_warmup_position(rules, attempt_seeds.next());
      if (!is_terminal(state) && !legal_moves(state).empty()) {
        generated = std::move(state);
        break;
      }
    }
    if (!generated.has_value()) {
      throw std::runtime_error(
          "could not generate a non-terminal arena warm-up position");
    }

    const std::vector<Move> moves = legal_moves(*generated);
    const Move candidate_move = candidate->choose_move(*generated);
    if (!contains_move(moves, candidate_move)) {
      throw std::logic_error(
          "candidate bot returned an illegal move during arena warm-up");
    }
    const Move reference_move = reference->choose_move(*generated);
    if (!contains_move(moves, reference_move)) {
      throw std::logic_error(
          "reference bot returned an illegal move during arena warm-up");
    }
  }
}

PositionEvaluation evaluate_position(const ArenaBotConfig &config, Entrant entrant,
                                     std::uint64_t seed,
                                     const GameState &state) {
  std::unique_ptr<Bot> bot = make_arena_bot(config, seed);
  DecisionReport decision = choose_and_measure(*bot, entrant, 1, state);
  return PositionEvaluation{entrant,
                            seed,
                            decision.move,
                            decision.elapsed_ns,
                            decision.stats,
                            decision.alpha_beta_stats,
                            decision.rank5_derived_stats,
                            decision.deep_turn_search_stats,
                            decision.deep_turn_search_profile_nodes};
}

std::uint64_t median_unsigned(std::vector<std::uint64_t> values) {
  if (values.empty()) {
    return 0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) {
    return values[middle];
  }
  const std::uint64_t lower = values[middle - 1];
  const std::uint64_t upper = values[middle];
  return lower + (upper - lower) / 2;
}

double median_double(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) {
    return values[middle];
  }
  return (values[middle - 1] + values[middle]) / 2.0;
}

std::uint64_t nearest_rank_percentile(
    const std::vector<std::uint64_t> &sorted_values, double fraction) {
  if (sorted_values.empty()) {
    return 0;
  }
  const std::size_t index =
      static_cast<std::size_t>(std::ceil(fraction * sorted_values.size())) - 1;
  return sorted_values[std::min(index, sorted_values.size() - 1)];
}

TimingSummary summarize_timing(
    const std::vector<const DecisionReport *> &decisions) {
  TimingSummary summary;
  summary.decisions = decisions.size();
  if (decisions.empty()) {
    return summary;
  }

  std::vector<std::uint64_t> elapsed;
  std::vector<double> throughput;
  std::vector<double> rollout_throughput;
  std::vector<double> node_throughput;
  elapsed.reserve(decisions.size());
  throughput.reserve(decisions.size());
  rollout_throughput.reserve(decisions.size());
  node_throughput.reserve(decisions.size());
  for (const DecisionReport *decision : decisions) {
    elapsed.push_back(decision->elapsed_ns);
    summary.total_ns += decision->elapsed_ns;
    if (decision->stats.has_value() && decision->elapsed_ns > 0) {
      throughput.push_back(
          static_cast<double>(decision->stats->iterations) * 1'000'000'000.0 /
          static_cast<double>(decision->elapsed_ns));
      rollout_throughput.push_back(
          static_cast<double>(decision->stats->simulated_plies) *
          1'000'000'000.0 / static_cast<double>(decision->elapsed_ns));
      node_throughput.push_back(
          static_cast<double>(decision->stats->nodes) * 1'000'000'000.0 /
          static_cast<double>(decision->elapsed_ns));
    } else if (decision->alpha_beta_stats.has_value() &&
               decision->elapsed_ns > 0) {
      node_throughput.push_back(
          static_cast<double>(decision->alpha_beta_stats->nodes) *
          1'000'000'000.0 / static_cast<double>(decision->elapsed_ns));
    } else if (decision->rank5_derived_stats.has_value() &&
               !decision->rank5_derived_stats->cached_continuation &&
               decision->elapsed_ns > 0) {
      node_throughput.push_back(
          static_cast<double>(decision->rank5_derived_stats->nodes) *
          1'000'000'000.0 / static_cast<double>(decision->elapsed_ns));
    } else if (decision->deep_turn_search_stats.has_value() &&
               !decision->deep_turn_search_stats->cached_continuation &&
               decision->elapsed_ns > 0) {
      node_throughput.push_back(
          static_cast<double>(decision->deep_turn_search_stats->nodes) *
          1'000'000'000.0 / static_cast<double>(decision->elapsed_ns));
    }
  }
  std::sort(elapsed.begin(), elapsed.end());
  summary.min_ns = elapsed.front();
  summary.median_ns = median_unsigned(elapsed);
  summary.p90_ns = nearest_rank_percentile(elapsed, 0.90);
  summary.p95_ns = nearest_rank_percentile(elapsed, 0.95);
  summary.p99_ns = nearest_rank_percentile(elapsed, 0.99);
  summary.max_ns = elapsed.back();
  summary.median_iterations_per_second = median_double(std::move(throughput));
  summary.median_simulated_plies_per_second =
      median_double(std::move(rollout_throughput));
  summary.median_nodes_per_second = median_double(std::move(node_throughput));
  return summary;
}

MctsSummary summarize_mcts(
    const std::vector<const DecisionReport *> &decisions) {
  MctsSummary summary;
  for (const DecisionReport *decision : decisions) {
    if (!decision->stats.has_value()) {
      continue;
    }
    const SearchStats &stats = *decision->stats;
    ++summary.searches;
    summary.iterations += stats.iterations;
    summary.nodes_sum += stats.nodes;
    summary.simulated_plies += stats.simulated_plies;
    summary.total_root_visits_sum += stats.total_root_visits;
    summary.reused_visits_sum += stats.reused_visits;
    summary.max_depth = std::max(summary.max_depth, stats.max_depth);
    summary.proven_nodes_sum += stats.proven_nodes;
    summary.proven_searches += stats.proven_winner.has_value() ? 1U : 0U;
    summary.tactical_probes += stats.tactical_probes;
    summary.tactical_nodes += stats.tactical_nodes;
    summary.tactical_solved_positions += stats.tactical_solved_positions;
    summary.tactical_depth_cutoffs += stats.tactical_depth_cutoffs;
    summary.tactical_node_cutoffs += stats.tactical_node_cutoffs;
    summary.max_tactical_depth =
        std::max(summary.max_tactical_depth, stats.max_tactical_depth);
    summary.rebuild_count_max =
        std::max(summary.rebuild_count_max, stats.rebuild_count);
    summary.expansion_saturated_searches += stats.expansion_saturated ? 1U : 0U;
  }
  return summary;
}

AlphaBetaSummary summarize_alpha_beta(
    const std::vector<const DecisionReport *> &decisions) {
  AlphaBetaSummary summary;
  for (const DecisionReport *decision : decisions) {
    if (!decision->alpha_beta_stats.has_value()) {
      continue;
    }
    const AlphaBetaSearchStats &stats = *decision->alpha_beta_stats;
    ++summary.searches;
    summary.nodes_sum += stats.nodes;
    summary.leaf_evaluations_sum += stats.leaf_evaluations;
    summary.terminal_nodes_sum += stats.terminal_nodes;
    summary.cutoffs_sum += stats.cutoffs;
    summary.transposition_probes_sum += stats.transposition_probes;
    summary.transposition_hits_sum += stats.transposition_hits;
    summary.transposition_cutoffs_sum += stats.transposition_cutoffs;
    summary.transposition_stores_sum += stats.transposition_stores;
    summary.physical_ply_cutoffs_sum += stats.physical_ply_cutoffs;
    summary.max_completed_turn_depth = std::max(
        summary.max_completed_turn_depth, stats.completed_turn_depth);
    summary.max_attempted_turn_depth = std::max(
        summary.max_attempted_turn_depth, stats.attempted_turn_depth);
    summary.max_physical_ply =
        std::max(summary.max_physical_ply, stats.max_physical_ply);
    summary.budget_exhausted_searches += stats.budget_exhausted ? 1U : 0U;
    ++summary.completed_turn_depth_histogram[stats.completed_turn_depth];
    ++summary.attempted_turn_depth_histogram[stats.attempted_turn_depth];
  }
  return summary;
}

Rank5DerivedSummary summarize_complete_turn_search(
    const std::vector<const DecisionReport *> &decisions,
    std::uint64_t requested_nodes,
    std::optional<CompleteTurnSearchStats> DecisionReport::*stats_member,
    std::string_view label) {
  Rank5DerivedSummary summary;
  std::vector<const DecisionReport *> all_edges;
  std::vector<const DecisionReport *> fresh_roots;
  all_edges.reserve(decisions.size());
  fresh_roots.reserve(decisions.size());
  for (const DecisionReport *decision : decisions) {
    const auto &maybe_stats = decision->*stats_member;
    if (!maybe_stats.has_value()) {
      continue;
    }
    const CompleteTurnSearchStats &stats = *maybe_stats;
    ++summary.decisions;
    all_edges.push_back(decision);
    summary.maximum_current_edge_index =
        std::max(summary.maximum_current_edge_index, stats.current_edge_index);
    if (stats.cached_continuation) {
      ++summary.cached_continuation_edges;
      continue;
    }
    ++summary.fresh_root_searches;
    summary.requested_nodes_sum += requested_nodes;
    summary.visited_nodes_sum += stats.nodes;
    summary.budget_exhausted_fresh_searches +=
        stats.budget_exhausted ? 1U : 0U;
    ++summary.completed_depth_histogram[stats.completed_turn_depth];
    ++summary.attempted_depth_histogram[stats.attempted_turn_depth];
    ++summary.planned_action_length_histogram[stats.planned_action_length];
    summary.minimum_root_score = summary.minimum_root_score.has_value()
                                     ? std::min(*summary.minimum_root_score,
                                                stats.root_score)
                                     : stats.root_score;
    summary.maximum_root_score = summary.maximum_root_score.has_value()
                                     ? std::max(*summary.maximum_root_score,
                                                stats.root_score)
                                     : stats.root_score;
    add_rank5_counter(summary.leaf_evaluations, stats.leaf_evaluations);
    add_rank5_counter(summary.terminal_nodes, stats.terminal_nodes);
    add_rank5_counter(summary.completed_actions, stats.completed_actions);
    add_rank5_counter(summary.cutoffs, stats.cutoffs);
    add_rank5_counter(summary.transposition_probes,
                      stats.transposition_probes);
    add_rank5_counter(summary.transposition_hits, stats.transposition_hits);
    add_rank5_counter(summary.transposition_cutoffs,
                      stats.transposition_cutoffs);
    add_rank5_counter(summary.transposition_stores,
                      stats.transposition_stores);
    add_rank5_counter(summary.continuation_transposition_hits,
                      stats.continuation_transposition_hits);
    add_rank5_counter(summary.evaluation_cache_probes,
                      stats.evaluation_cache_probes);
    add_rank5_counter(summary.evaluation_cache_hits,
                      stats.evaluation_cache_hits);
    add_rank5_counter(summary.terminal_bound_cutoffs,
                      stats.terminal_bound_cutoffs);
    add_rank5_counter(summary.forced_edges, stats.forced_edges);
    add_rank5_counter(summary.root_seed_actions, stats.root_seed_actions);
    add_rank5_counter(summary.root_transposition_reuses,
                      stats.root_transposition_reuses);
    add_rank5_counter(summary.max_action_edges, stats.max_action_edges);
    fresh_roots.push_back(decision);
  }
  summary.fresh_root_timing = summarize_timing(fresh_roots);
  summary.all_edge_timing = summarize_timing(all_edges);
  if (summary.decisions !=
          summary.fresh_root_searches + summary.cached_continuation_edges ||
      summary.fresh_root_timing.decisions != summary.fresh_root_searches ||
      summary.all_edge_timing.decisions != summary.decisions) {
    throw std::logic_error("inconsistent " + std::string(label) +
                           " arena summary");
  }
  return summary;
}

Rank5DerivedSummary summarize_rank5_derived(
    const std::vector<const DecisionReport *> &decisions) {
  return summarize_complete_turn_search(
      decisions, Rank5DerivedConfig::profile_max_nodes,
      &DecisionReport::rank5_derived_stats, "Rank5Derived");
}

Rank5DerivedSummary summarize_deep_turn_search(
    const std::vector<const DecisionReport *> &decisions,
    std::uint64_t requested_nodes) {
  return summarize_complete_turn_search(
      decisions, requested_nodes, &DecisionReport::deep_turn_search_stats,
      "DeepTurnSearch");
}

Record record_for(const std::vector<GameReport> &games, Entrant entrant,
                  std::optional<Player> color) {
  Record record;
  for (const GameReport &game : games) {
    const Player entrant_color = game.player_one.entrant == entrant
                                     ? Player::One
                                     : Player::Two;
    if (color.has_value() && entrant_color != *color) {
      continue;
    }
    ++record.games;
    if (game.truncated) {
      ++record.truncations;
    } else if (game.winning_entrant == entrant) {
      ++record.wins;
    } else {
      ++record.losses;
    }
  }
  return record;
}

std::vector<const DecisionReport *> decisions_for(
    const std::vector<GameReport> &games, Entrant entrant) {
  std::vector<const DecisionReport *> result;
  for (const GameReport &game : games) {
    for (const DecisionReport &decision : game.decisions) {
      if (decision.entrant == entrant) {
        result.push_back(&decision);
      }
    }
  }
  return result;
}

std::vector<const DecisionReport *> decisions_for(
    const std::vector<PositionReport> &positions, Entrant entrant,
    std::vector<DecisionReport> &storage) {
  storage.reserve(positions.size());
  for (const PositionReport &position : positions) {
    const PositionEvaluation &evaluation =
        entrant == Entrant::Candidate ? position.candidate : position.reference;
    storage.push_back(DecisionReport{
        1, entrant, position.state.to_move, position.state.ball, evaluation.move,
        evaluation.elapsed_ns, evaluation.stats, evaluation.alpha_beta_stats,
        evaluation.rank5_derived_stats, evaluation.deep_turn_search_stats,
        evaluation.deep_turn_search_profile_nodes});
  }
  std::vector<const DecisionReport *> result;
  result.reserve(storage.size());
  for (const DecisionReport &decision : storage) {
    result.push_back(&decision);
  }
  return result;
}

std::vector<std::optional<double>> candidate_pair_scores(
    const std::vector<GameReport> &games, std::size_t pair_count) {
  std::vector<double> completed_scores(pair_count, 0.0);
  std::vector<std::size_t> games_per_pair(pair_count, 0);
  std::vector<bool> invalid(pair_count, false);
  for (const GameReport &game : games) {
    if (game.pair_index >= pair_count) {
      throw std::logic_error("arena game has an out-of-range pair index");
    }
    ++games_per_pair[game.pair_index];
    if (game.truncated) {
      invalid[game.pair_index] = true;
    } else if (game.winning_entrant == Entrant::Candidate) {
      completed_scores[game.pair_index] += 0.5;
    }
  }
  std::vector<std::optional<double>> scores(pair_count);
  for (std::size_t pair = 0; pair < pair_count; ++pair) {
    if (!invalid[pair] && games_per_pair[pair] == 2) {
      scores[pair] = completed_scores[pair];
    }
  }
  return scores;
}

BootstrapInterval bootstrap_interval(
    const std::vector<std::optional<double>> &pair_scores,
    std::uint64_t base_seed, std::size_t samples) {
  BootstrapInterval interval;
  interval.seed = base_seed ^ kBootstrapSeedSalt;
  interval.samples = samples;
  std::vector<double> valid_scores;
  valid_scores.reserve(pair_scores.size());
  for (const std::optional<double> score : pair_scores) {
    if (score.has_value()) {
      valid_scores.push_back(*score);
    }
  }
  interval.valid_pairs = valid_scores.size();
  interval.invalid_pairs = pair_scores.size() - valid_scores.size();
  if (valid_scores.empty()) {
    return interval;
  }
  SplitMix64 random{interval.seed};
  std::vector<double> means;
  means.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    double total = 0.0;
    for (std::size_t draw = 0; draw < valid_scores.size(); ++draw) {
      total += valid_scores[random.index(valid_scores.size())];
    }
    means.push_back(100.0 * total / static_cast<double>(valid_scores.size()));
  }
  std::sort(means.begin(), means.end());
  const std::size_t lower = static_cast<std::size_t>(
      std::floor(0.025 * static_cast<double>(samples - 1)));
  const std::size_t upper = static_cast<std::size_t>(
      std::ceil(0.975 * static_cast<double>(samples - 1)));
  interval.lower_percent = means[lower];
  interval.upper_percent = means[upper];
  return interval;
}

}  // namespace papersoccer::arena::detail
