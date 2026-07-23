#include "internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "papersoccer/rules.hpp"

namespace papersoccer::arena::detail {

using Clock = std::chrono::steady_clock;

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
                        alpha_beta_search_stats(bot)};
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
      return OpeningReport{pair_index, generation_seed, attempt + 1,
                           std::move(state)};
    }
  }
  throw std::runtime_error(
      "could not generate a non-terminal arena opening at the requested ply");
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
                            decision.alpha_beta_stats};
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
    }
  }
  std::sort(elapsed.begin(), elapsed.end());
  summary.min_ns = elapsed.front();
  summary.median_ns = median_unsigned(elapsed);
  const std::size_t p95_index =
      static_cast<std::size_t>(std::ceil(0.95 * elapsed.size())) - 1;
  summary.p95_ns = elapsed[p95_index];
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
        evaluation.elapsed_ns, evaluation.stats, evaluation.alpha_beta_stats});
  }
  std::vector<const DecisionReport *> result;
  result.reserve(storage.size());
  for (const DecisionReport &decision : storage) {
    result.push_back(&decision);
  }
  return result;
}

std::vector<double> candidate_pair_scores(const std::vector<GameReport> &games,
                                          std::size_t pair_count) {
  std::vector<double> scores(pair_count, 0.0);
  for (const GameReport &game : games) {
    if (game.truncated) {
      scores[game.pair_index] += 0.25;
    } else if (game.winning_entrant == Entrant::Candidate) {
      scores[game.pair_index] += 0.5;
    }
  }
  return scores;
}

BootstrapInterval bootstrap_interval(const std::vector<double> &pair_scores,
                                     std::uint64_t base_seed,
                                     std::size_t samples) {
  BootstrapInterval interval;
  interval.seed = base_seed ^ kBootstrapSeedSalt;
  interval.samples = samples;
  SplitMix64 random{interval.seed};
  std::vector<double> means;
  means.reserve(samples);
  for (std::size_t sample = 0; sample < samples; ++sample) {
    double total = 0.0;
    for (std::size_t draw = 0; draw < pair_scores.size(); ++draw) {
      total += pair_scores[random.index(pair_scores.size())];
    }
    means.push_back(100.0 * total / static_cast<double>(pair_scores.size()));
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
