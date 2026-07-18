#include "internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

namespace papersoccer::arena {

namespace detail {

void write_string(std::ostream &out, std::string_view value) {
  out << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (character < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(character);
        }
    }
  }
  out << '"';
}

void write_bool(std::ostream &out, bool value) { out << (value ? "true" : "false"); }

void write_uint64_string(std::ostream &out, std::uint64_t value) {
  write_string(out, std::to_string(value));
}

void write_point(std::ostream &out, Point point) {
  out << "{\"x\":" << point.x << ",\"y\":" << point.y << '}';
}

void write_bot_config(std::ostream &out, const ArenaBotConfig &config) {
  out << "{\"kind\":";
  write_string(out, kind_name(config.kind));
  switch (config.kind) {
    case BotKind::Random:
      break;
    case BotKind::Mcts:
      out << ",\"iterations\":" << config.iterations
          << ",\"exploration\":" << config.exploration
          << ",\"rollout_policy\":";
      write_string(out, policy_name(config.rollout_policy));
      out << ",\"leaf_policy\":";
      write_string(out, leaf_policy_name(config.leaf_policy));
      out << ",\"quiescence_max_depth\":" << config.quiescence_max_depth
          << ",\"quiescence_max_nodes\":" << config.quiescence_max_nodes;
      out << ",\"reuse_tree\":";
      write_bool(out, config.reuse_tree);
      out << ",\"max_nodes\":" << config.max_nodes;
      break;
    case BotKind::AlphaBeta:
      out << ",\"max_turn_depth\":" << config.alpha_beta_depth
          << ",\"max_nodes\":" << config.alpha_beta_max_nodes
          << ",\"transposition_table_entries\":"
          << config.alpha_beta_transposition_table_entries
          << ",\"max_search_plies\":"
          << config.alpha_beta_max_search_plies;
      break;
  }
  out << '}';
}

void write_stats(std::ostream &out, const SearchStats &stats) {
  out << "{\"iterations\":" << stats.iterations << ",\"nodes\":" << stats.nodes
      << ",\"simulated_plies\":" << stats.simulated_plies
      << ",\"root_value\":" << stats.root_value
      << ",\"total_root_visits\":" << stats.total_root_visits
      << ",\"reused_visits\":" << stats.reused_visits
      << ",\"max_depth\":" << stats.max_depth
      << ",\"proven_nodes\":" << stats.proven_nodes
      << ",\"proven_winner\":";
  if (stats.proven_winner.has_value()) {
    write_string(out, player_name(*stats.proven_winner));
  } else {
    out << "null";
  }
  out << ",\"tactical_probes\":" << stats.tactical_probes
      << ",\"tactical_nodes\":" << stats.tactical_nodes
      << ",\"tactical_solved_positions\":"
      << stats.tactical_solved_positions
      << ",\"tactical_depth_cutoffs\":" << stats.tactical_depth_cutoffs
      << ",\"tactical_node_cutoffs\":" << stats.tactical_node_cutoffs
      << ",\"max_tactical_depth\":" << stats.max_tactical_depth
      << ",\"rebuild_count\":" << stats.rebuild_count
      << ",\"expansion_saturated\":";
  write_bool(out, stats.expansion_saturated);
  out << '}';
}

void write_alpha_beta_stats(std::ostream &out,
                            const AlphaBetaSearchStats &stats) {
  out << "{\"completed_turn_depth\":" << stats.completed_turn_depth
      << ",\"attempted_turn_depth\":" << stats.attempted_turn_depth
      << ",\"nodes\":" << stats.nodes
      << ",\"leaf_evaluations\":" << stats.leaf_evaluations
      << ",\"terminal_nodes\":" << stats.terminal_nodes
      << ",\"cutoffs\":" << stats.cutoffs
      << ",\"transposition_probes\":" << stats.transposition_probes
      << ",\"transposition_hits\":" << stats.transposition_hits
      << ",\"transposition_cutoffs\":" << stats.transposition_cutoffs
      << ",\"transposition_stores\":" << stats.transposition_stores
      << ",\"physical_ply_cutoffs\":" << stats.physical_ply_cutoffs
      << ",\"max_physical_ply\":" << stats.max_physical_ply
      << ",\"root_score\":" << stats.root_score
      << ",\"budget_exhausted\":";
  write_bool(out, stats.budget_exhausted);
  out << ",\"principal_variation\":[";
  for (std::size_t index = 0; index < stats.principal_variation.size();
       ++index) {
    if (index != 0) {
      out << ',';
    }
    write_point(out, stats.principal_variation[index].to);
  }
  out << "],\"root_moves\":[";
  for (std::size_t index = 0; index < stats.root_moves.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << "{\"move\":";
    write_point(out, stats.root_moves[index].move.to);
    out << ",\"score\":" << stats.root_moves[index].score
        << ",\"bound\":";
    write_string(out, alpha_beta_bound_name(stats.root_moves[index].bound));
    out << '}';
  }
  out << "]}";
}

void write_participant(std::ostream &out, const Participant &participant) {
  out << "{\"bot\":";
  write_string(out, entrant_name(participant.entrant));
  out << ",\"seed\":";
  write_uint64_string(out, participant.seed);
  out << ",\"config\":";
  write_bot_config(out, participant.config);
  out << '}';
}

void write_decision(std::ostream &out, const DecisionReport &decision) {
  out << "{\"ply\":" << decision.ply << ",\"bot\":";
  write_string(out, entrant_name(decision.entrant));
  out << ",\"player\":";
  write_string(out, player_name(decision.player));
  out << ",\"from\":";
  write_point(out, decision.from);
  out << ",\"to\":";
  write_point(out, decision.move.to);
  out << ",\"elapsed_ns\":" << decision.elapsed_ns << ",\"legal\":true,\"mcts\":";
  if (decision.stats.has_value()) {
    write_stats(out, *decision.stats);
  } else {
    out << "null";
  }
  out << ",\"alpha_beta\":";
  if (decision.alpha_beta_stats.has_value()) {
    write_alpha_beta_stats(out, *decision.alpha_beta_stats);
  } else {
    out << "null";
  }
  out << '}';
}

void write_opening(std::ostream &out, const OpeningReport &opening,
                   std::size_t requested_plies) {
  const std::size_t actual_plies = opening.state.path.size() - 1;
  out << "{\"pair_index\":" << opening.pair_index
      << ",\"generation_seed\":";
  write_uint64_string(out, opening.generation_seed);
  out << ",\"attempts\":" << opening.attempts
      << ",\"requested_plies\":" << requested_plies
      << ",\"actual_plies\":" << actual_plies << ",\"moves\":[";
  for (std::size_t index = 1; index < opening.state.path.size(); ++index) {
    if (index != 1) {
      out << ',';
    }
    write_point(out, opening.state.path[index]);
  }
  out << "],\"state\":{\"ball\":";
  write_point(out, opening.state.ball);
  out << ",\"to_move\":";
  write_string(out, player_name(opening.state.to_move));
  out << "}}";
}

void write_record(std::ostream &out, const Record &record) {
  const double percent =
      record.games == 0 ? 0.0 : 100.0 * record.score() / record.games;
  out << "{\"games\":" << record.games << ",\"wins\":" << record.wins
      << ",\"losses\":" << record.losses
      << ",\"truncations\":" << record.truncations
      << ",\"score_points\":" << record.score()
      << ",\"score_percent\":" << percent << '}';
}

void write_timing(std::ostream &out, const TimingSummary &summary) {
  out << "{\"decisions\":" << summary.decisions
      << ",\"total_ns\":" << summary.total_ns
      << ",\"min_ns\":" << summary.min_ns
      << ",\"median_ns\":" << summary.median_ns
      << ",\"p95_ns\":" << summary.p95_ns
      << ",\"max_ns\":" << summary.max_ns
      << ",\"median_iterations_per_second\":"
      << summary.median_iterations_per_second
      << ",\"median_simulated_plies_per_second\":"
      << summary.median_simulated_plies_per_second
      << ",\"median_nodes_per_second\":" << summary.median_nodes_per_second
      << '}';
}

void write_mcts_summary(std::ostream &out, const MctsSummary &summary) {
  const double tactical_solution_rate =
      summary.tactical_probes == 0
          ? 0.0
          : static_cast<double>(summary.tactical_solved_positions) /
                static_cast<double>(summary.tactical_probes);
  out << "{\"searches\":" << summary.searches
      << ",\"iterations\":" << summary.iterations
      << ",\"nodes_sum\":" << summary.nodes_sum
      << ",\"simulated_plies\":" << summary.simulated_plies
      << ",\"total_root_visits_sum\":" << summary.total_root_visits_sum
      << ",\"reused_visits_sum\":" << summary.reused_visits_sum
      << ",\"max_depth\":" << summary.max_depth
      << ",\"proven_nodes_sum\":" << summary.proven_nodes_sum
      << ",\"proven_searches\":" << summary.proven_searches
      << ",\"tactical_probes\":" << summary.tactical_probes
      << ",\"tactical_nodes\":" << summary.tactical_nodes
      << ",\"tactical_solved_positions\":"
      << summary.tactical_solved_positions
      << ",\"tactical_solution_rate\":" << tactical_solution_rate
      << ",\"tactical_depth_cutoffs\":" << summary.tactical_depth_cutoffs
      << ",\"tactical_node_cutoffs\":" << summary.tactical_node_cutoffs
      << ",\"max_tactical_depth\":" << summary.max_tactical_depth
      << ",\"rebuild_count_max\":" << summary.rebuild_count_max
      << ",\"expansion_saturated_searches\":"
      << summary.expansion_saturated_searches << '}';
}

void write_alpha_beta_summary(std::ostream &out,
                              const AlphaBetaSummary &summary) {
  const double transposition_hit_rate =
      summary.transposition_probes_sum == 0
          ? 0.0
          : static_cast<double>(summary.transposition_hits_sum) /
                static_cast<double>(summary.transposition_probes_sum);
  out << "{\"searches\":" << summary.searches
      << ",\"nodes_sum\":" << summary.nodes_sum
      << ",\"leaf_evaluations_sum\":" << summary.leaf_evaluations_sum
      << ",\"terminal_nodes_sum\":" << summary.terminal_nodes_sum
      << ",\"cutoffs_sum\":" << summary.cutoffs_sum
      << ",\"transposition_probes_sum\":"
      << summary.transposition_probes_sum
      << ",\"transposition_hits_sum\":" << summary.transposition_hits_sum
      << ",\"transposition_hit_rate\":" << transposition_hit_rate
      << ",\"transposition_cutoffs_sum\":"
      << summary.transposition_cutoffs_sum
      << ",\"transposition_stores_sum\":"
      << summary.transposition_stores_sum
      << ",\"physical_ply_cutoffs_sum\":"
      << summary.physical_ply_cutoffs_sum
      << ",\"max_completed_turn_depth\":"
      << summary.max_completed_turn_depth
      << ",\"max_attempted_turn_depth\":"
      << summary.max_attempted_turn_depth
      << ",\"max_physical_ply\":" << summary.max_physical_ply
      << ",\"budget_exhausted_searches\":"
      << summary.budget_exhausted_searches << '}';
}

void write_entrant_summary(std::ostream &out, Entrant entrant,
                           const std::vector<GameReport> &games) {
  const std::vector<const DecisionReport *> decisions = decisions_for(games, entrant);
  out << "{\"overall\":";
  write_record(out, record_for(games, entrant));
  out << ",\"color_splits\":{\"player_one\":";
  write_record(out, record_for(games, entrant, Player::One));
  out << ",\"player_two\":";
  write_record(out, record_for(games, entrant, Player::Two));
  out << "},\"timing\":";
  write_timing(out, summarize_timing(decisions));
  out << ",\"mcts\":";
  write_mcts_summary(out, summarize_mcts(decisions));
  out << ",\"alpha_beta\":";
  write_alpha_beta_summary(out, summarize_alpha_beta(decisions));
  out << '}';
}

void write_position_entrant_summary(std::ostream &out, Entrant entrant,
                                    const std::vector<PositionReport> &positions) {
  std::vector<DecisionReport> storage;
  const std::vector<const DecisionReport *> decisions =
      decisions_for(positions, entrant, storage);
  out << "{\"timing\":";
  write_timing(out, summarize_timing(decisions));
  out << ",\"mcts\":";
  write_mcts_summary(out, summarize_mcts(decisions));
  out << ",\"alpha_beta\":";
  write_alpha_beta_summary(out, summarize_alpha_beta(decisions));
  out << '}';
}

}  // namespace detail

std::string run_matches_json(const MatchesConfig &config) {
  using namespace detail;
  validate_common(config.rules, config.candidate, config.reference);
  if (config.seed_pairs == 0) {
    throw std::invalid_argument("arena seed pair count must be greater than zero");
  }
  if (config.max_plies == 0) {
    throw std::invalid_argument("arena max plies must be greater than zero");
  }
  if (config.opening_plies >= config.max_plies && config.opening_plies != 0) {
    throw std::invalid_argument(
        "arena opening plies must be less than max plies");
  }
  if (config.bootstrap_samples == 0) {
    throw std::invalid_argument("arena bootstrap samples must be greater than zero");
  }
  if (config.seed_pairs > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::invalid_argument("arena seed pair count is too large");
  }

  SplitMix64 seeds{config.base_seed};
  SplitMix64 opening_pair_seeds{config.base_seed ^ kOpeningSeedSalt};
  std::vector<OpeningReport> openings;
  if (config.opening_plies != 0) {
    openings.reserve(config.seed_pairs);
  }
  std::vector<GameReport> games;
  games.reserve(config.seed_pairs * 2);
  for (std::size_t pair = 0; pair < config.seed_pairs; ++pair) {
    const std::uint64_t candidate_seed = seeds.next();
    const std::uint64_t reference_seed = seeds.next();
    GameState initial_state = make_initial_state(config.rules);
    if (config.opening_plies != 0) {
      OpeningReport opening = generate_opening(
          pair, config.rules, opening_pair_seeds.next(), config.opening_plies);
      initial_state = opening.state;
      openings.push_back(std::move(opening));
    }
    games.push_back(play_game(
        pair, 0,
        Participant{Entrant::Candidate, Player::One, candidate_seed,
                    config.candidate},
        Participant{Entrant::Reference, Player::Two, reference_seed,
                    config.reference},
        initial_state, config.max_plies));
    games.push_back(play_game(
        pair, 1,
        Participant{Entrant::Reference, Player::One, reference_seed,
                    config.reference},
        Participant{Entrant::Candidate, Player::Two, candidate_seed,
                    config.candidate},
        initial_state, config.max_plies));
  }

  const std::vector<double> pair_scores =
      candidate_pair_scores(games, config.seed_pairs);
  const BootstrapInterval interval = bootstrap_interval(
      pair_scores, config.base_seed, config.bootstrap_samples);
  const std::size_t truncations = static_cast<std::size_t>(std::count_if(
      games.begin(), games.end(), [](const GameReport &game) { return game.truncated; }));

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.arena.v1\",\"mode\":\"matches\","
         "\"runtime\":";
  write_string(out, runtime_name());
  out << ",\"timing_unit\":\"nanoseconds\","
         "\"configuration\":{\"rules\":{\"width\":"
      << config.rules.width << ",\"height\":" << config.rules.height
      << "},\"base_seed\":";
  write_uint64_string(out, config.base_seed);
  out << ",\"seed_derivation\":\"splitmix64\""
      << ",\"seed_pairs\":" << config.seed_pairs
      << ",\"games\":" << games.size()
      << ",\"opening_plies\":" << config.opening_plies
      << ",\"opening_generator\":\"uniform_random\""
      << ",\"opening_seed_derivation\":"
         "\"domain_separated_splitmix64\""
      << ",\"max_plies\":" << config.max_plies
      << ",\"bootstrap_samples\":" << config.bootstrap_samples
      << ",\"candidate\":";
  write_bot_config(out, config.candidate);
  out << ",\"reference\":";
  write_bot_config(out, config.reference);
  out << "},\"openings\":[";
  for (std::size_t index = 0; index < openings.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    write_opening(out, openings[index], config.opening_plies);
  }
  out << "],\"games\":[";
  for (std::size_t index = 0; index < games.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    const GameReport &game = games[index];
    out << "{\"pair_index\":" << game.pair_index
        << ",\"game_in_pair\":" << game.game_in_pair
        << ",\"player_one\":";
    write_participant(out, game.player_one);
    out << ",\"player_two\":";
    write_participant(out, game.player_two);
    out << ",\"outcome\":{\"status\":";
    write_string(out, status_name(game.status));
    out << ",\"winner\":";
    if (game.winning_entrant.has_value()) {
      write_string(out, entrant_name(*game.winning_entrant));
    } else {
      out << "null";
    }
    out << ",\"reason\":";
    write_string(out, game.truncated ? "ply_limit" : "terminal");
    out << ",\"truncated\":";
    write_bool(out, game.truncated);
    out << ",\"plies\":" << game.plies << "},\"decisions\":[";
    for (std::size_t decision_index = 0;
         decision_index < game.decisions.size(); ++decision_index) {
      if (decision_index != 0) {
        out << ',';
      }
      write_decision(out, game.decisions[decision_index]);
    }
    out << "]}";
  }
  out << "],\"summary\":{\"illegal_moves\":0,\"truncations\":" << truncations
      << ",\"candidate\":";
  write_entrant_summary(out, Entrant::Candidate, games);
  out << ",\"reference\":";
  write_entrant_summary(out, Entrant::Reference, games);
  out << ",\"pair_scores\":[";
  for (std::size_t index = 0; index < pair_scores.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << pair_scores[index];
  }
  out << "],\"pair_bootstrap_95\":{\"method\":"
         "\"deterministic_pair_resampling\",\"seed\":";
  write_uint64_string(out, interval.seed);
  out << ",\"samples\":" << interval.samples
      << ",\"confidence\":0.95,\"lower_percent\":"
      << interval.lower_percent << ",\"upper_percent\":"
      << interval.upper_percent << "}}}";
  return out.str();
}

std::string run_positions_json(const PositionsConfig &config) {
  using namespace detail;
  validate_common(config.rules, config.candidate, config.reference);
  if (config.position_count == 0) {
    throw std::invalid_argument("arena position count must be greater than zero");
  }

  SplitMix64 seeds{config.base_seed};
  std::vector<PositionReport> positions;
  positions.reserve(config.position_count);
  for (std::size_t index = 0; index < config.position_count; ++index) {
    std::optional<GameState> generated;
    std::uint64_t generation_seed = 0;
    for (std::size_t attempt = 0; attempt < kMaxPositionGenerationAttempts;
         ++attempt) {
      generation_seed = seeds.next();
      GameState state =
          generate_position(config.rules, generation_seed, config.generation_plies);
      if (!is_terminal(state) && !legal_moves(state).empty()) {
        generated = std::move(state);
        break;
      }
    }
    if (!generated.has_value()) {
      throw std::runtime_error(
          "could not generate a non-terminal arena position at the requested ply");
    }
    const std::uint64_t candidate_seed = seeds.next();
    const std::uint64_t reference_seed = seeds.next();
    PositionReport report;
    report.index = index;
    report.generation_seed = generation_seed;
    report.state = std::move(*generated);
    report.candidate = evaluate_position(
        config.candidate, Entrant::Candidate, candidate_seed, report.state);
    report.reference = evaluate_position(
        config.reference, Entrant::Reference, reference_seed, report.state);
    positions.push_back(std::move(report));
  }

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.arena.v1\",\"mode\":\"positions\","
         "\"runtime\":";
  write_string(out, runtime_name());
  out << ",\"timing_unit\":\"nanoseconds\","
         "\"configuration\":{\"rules\":{\"width\":"
      << config.rules.width << ",\"height\":" << config.rules.height
      << "},\"base_seed\":";
  write_uint64_string(out, config.base_seed);
  out << ",\"seed_derivation\":\"splitmix64\""
      << ",\"position_count\":" << config.position_count
      << ",\"generation_plies\":" << config.generation_plies
      << ",\"candidate\":";
  write_bot_config(out, config.candidate);
  out << ",\"reference\":";
  write_bot_config(out, config.reference);
  out << "},\"positions\":[";
  for (std::size_t index = 0; index < positions.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    const PositionReport &position = positions[index];
    out << "{\"index\":" << position.index
        << ",\"generation_seed\":";
    write_uint64_string(out, position.generation_seed);
    out << ",\"generated_plies\":" << config.generation_plies
        << ",\"state\":{\"ball\":";
    write_point(out, position.state.ball);
    out << ",\"to_move\":";
    write_string(out, player_name(position.state.to_move));
    out << ",\"used_segments\":" << position.state.used_segments.size()
        << ",\"visited_points\":" << position.state.visit_count.size()
        << "},\"evaluations\":[";
    const PositionEvaluation *evaluations[] = {&position.candidate,
                                               &position.reference};
    for (std::size_t evaluation_index = 0; evaluation_index < 2;
         ++evaluation_index) {
      if (evaluation_index != 0) {
        out << ',';
      }
      const PositionEvaluation &evaluation = *evaluations[evaluation_index];
      out << "{\"bot\":";
      write_string(out, entrant_name(evaluation.entrant));
      out << ",\"seed\":";
      write_uint64_string(out, evaluation.seed);
      out << ",\"move\":";
      write_point(out, evaluation.move.to);
      out << ",\"elapsed_ns\":" << evaluation.elapsed_ns
          << ",\"legal\":true,\"mcts\":";
      if (evaluation.stats.has_value()) {
        write_stats(out, *evaluation.stats);
      } else {
        out << "null";
      }
      out << ",\"alpha_beta\":";
      if (evaluation.alpha_beta_stats.has_value()) {
        write_alpha_beta_stats(out, *evaluation.alpha_beta_stats);
      } else {
        out << "null";
      }
      out << '}';
    }
    out << "]}";
  }
  out << "],\"summary\":{\"illegal_moves\":0,\"candidate\":";
  write_position_entrant_summary(out, Entrant::Candidate, positions);
  out << ",\"reference\":";
  write_position_entrant_summary(out, Entrant::Reference, positions);
  out << "}}";
  return out.str();
}

}  // namespace papersoccer::arena
