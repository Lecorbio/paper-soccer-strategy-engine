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

void write_optional_double(std::ostream &out,
                           const std::optional<double> &value) {
  if (value.has_value()) {
    out << *value;
  } else {
    out << "null";
  }
}

void write_point(std::ostream &out, Point point) {
  out << "{\"x\":" << point.x << ",\"y\":" << point.y << '}';
}

std::string_view rules_profile_name(const RulesConfig &rules) noexcept {
  if (rules.goal_rule == GoalRule::OpponentGoalOnly &&
      rules.blocked_rule == BlockedRule::PlayerToMoveLoses) {
    return "normal";
  }
  if (rules.goal_rule == GoalRule::OwnGoalsAllowed &&
      rules.blocked_rule == BlockedRule::MoverLoses) {
    return "codingame";
  }
  return "custom";
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
    case BotKind::JacekInspired:
      out << ",\"max_turn_depth\":" << config.alpha_beta_depth
          << ",\"max_nodes\":" << config.alpha_beta_max_nodes
          << ",\"transposition_table_entries\":"
          << config.alpha_beta_transposition_table_entries
          << ",\"max_search_plies\":"
          << config.alpha_beta_max_search_plies
          << ",\"model_sha256\":";
      write_string(out, JacekInspiredBot::model_sha256());
      break;
    case BotKind::Rank5Derived:
      out << ",\"profile\":";
      write_string(out, Rank5DerivedBot::profile_name());
      out << ",\"max_turn_depth\":"
          << Rank5DerivedConfig::maximum_turn_depth
          << ",\"max_nodes\":" << Rank5DerivedConfig::profile_max_nodes
          << ",\"transposition_table_entries\":65536"
          << ",\"evaluation_cache_entries\":32768"
          << ",\"max_time_ms\":0"
          << ",\"model_blend_percent\":0"
          << ",\"replay_corrections\":false"
          << ",\"replay_book_enabled\":false,\"original_sha256\":";
      write_string(out, Rank5DerivedBot::original_sha256());
      break;
    case BotKind::DeepTurnSearch: {
      const CompleteTurnAnalysisConfig profile =
          CompleteTurnAnalysisConfig::deep(config.complete_turn_max_nodes);
      out << ",\"profile\":";
      write_string(out, profile.profile_name());
      out << ",\"max_turn_depth\":" << profile.max_turn_depth
          << ",\"max_nodes\":" << profile.max_nodes
          << ",\"transposition_table_entries\":"
          << profile.transposition_table_entries
          << ",\"evaluation_cache_entries\":"
          << profile.evaluation_table_entries
          << ",\"max_time_ms\":0,\"model_blend_percent\":0"
          << ",\"replay_corrections\":false"
          << ",\"ranked_source_sha256\":";
      write_string(out, Rank5DerivedBot::original_sha256());
      break;
    }
    case BotKind::JacekReplayBfm: {
      const JacekReplayBfmConfig &bfm = config.jacek_replay_bfm;
      out << ",\"model_path\":";
      write_string(out, bfm.model_path);
      out << ",\"feature_schema\":";
      write_string(out, JacekReplayBfmBot::feature_schema());
      out << ",\"feature_schema_sha256\":";
      write_string(out, JacekReplayBfmBot::feature_schema_sha256());
      out << ",\"max_time_ms\":" << bfm.max_time_ms
          << ",\"max_tree_nodes\":" << bfm.max_tree_nodes
          << ",\"max_actions\":" << bfm.max_actions
          << ",\"max_partial_paths\":" << bfm.max_partial_paths
          << ",\"exploration\":" << bfm.exploration << ",\"fpu\":" << bfm.fpu;
      break;
    }
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

void write_complete_turn_stats(std::ostream &out,
                               const CompleteTurnSearchStats &stats,
                               std::uint64_t profile_node_budget) {
  const std::uint64_t requested_nodes =
      stats.cached_continuation ? 0 : profile_node_budget;
  out << "{\"profile_node_budget\":"
      << profile_node_budget
      << ",\"requested_nodes\":" << requested_nodes
      << ",\"visited_nodes\":" << stats.nodes
      << ",\"completed_turn_depth\":" << stats.completed_turn_depth
      << ",\"attempted_turn_depth\":" << stats.attempted_turn_depth
      << ",\"root_score\":" << stats.root_score
      << ",\"budget_exhausted\":";
  write_bool(out, stats.budget_exhausted);
  out << ",\"planned_action_length\":" << stats.planned_action_length
      << ",\"current_edge_index\":" << stats.current_edge_index
      << ",\"cached_continuation\":";
  write_bool(out, stats.cached_continuation);
  out << ",\"cached_moves_remaining\":" << stats.cached_moves_remaining
      << ",\"search_ordinal_in_game\":" << stats.searches
      << ",\"leaf_evaluations\":" << stats.leaf_evaluations
      << ",\"terminal_nodes\":" << stats.terminal_nodes
      << ",\"completed_actions\":" << stats.completed_actions
      << ",\"cutoffs\":" << stats.cutoffs
      << ",\"transposition_probes\":" << stats.transposition_probes
      << ",\"transposition_hits\":" << stats.transposition_hits
      << ",\"transposition_cutoffs\":" << stats.transposition_cutoffs
      << ",\"transposition_stores\":" << stats.transposition_stores
      << ",\"continuation_transposition_hits\":"
      << stats.continuation_transposition_hits
      << ",\"evaluation_cache_probes\":" << stats.evaluation_cache_probes
      << ",\"evaluation_cache_hits\":" << stats.evaluation_cache_hits
      << ",\"terminal_bound_cutoffs\":" << stats.terminal_bound_cutoffs
      << ",\"forced_edges\":" << stats.forced_edges
      << ",\"root_seed_actions\":" << stats.root_seed_actions
      << ",\"root_transposition_reuses\":"
      << stats.root_transposition_reuses
      << ",\"max_action_edges\":" << stats.max_action_edges << '}';
}

void write_rank5_derived_stats(std::ostream &out,
                               const Rank5DerivedSearchStats &stats) {
  write_complete_turn_stats(out, stats,
                            Rank5DerivedConfig::profile_max_nodes);
}

void write_jacek_replay_bfm_stats(std::ostream &out,
                                  const JacekReplayBfmSearchStats &stats,
                                  std::string_view model_sha256) {
  out << "{\"model_sha256\":";
  write_string(out, model_sha256);
  out << ",\"feature_schema\":";
  write_string(out, JacekReplayBfmBot::feature_schema());
  out << ",\"feature_schema_sha256\":";
  write_string(out, JacekReplayBfmBot::feature_schema_sha256());
  out << ",\"expansions\":" << stats.expansions
      << ",\"generated_actions\":" << stats.generated_actions
      << ",\"retained_actions\":" << stats.retained_actions
      << ",\"neural_evaluations\":" << stats.neural_evaluations
      << ",\"visits\":" << stats.visits
      << ",\"completed_actions\":" << stats.completed_actions
      << ",\"duplicate_boundaries\":" << stats.duplicate_boundaries
      << ",\"partial_paths\":" << stats.partial_paths
      << ",\"fifo_extractions\":" << stats.fifo_extractions
      << ",\"lifo_extractions\":" << stats.lifo_extractions
      << ",\"tactical_proofs\":" << stats.tactical_proofs
      << ",\"tactical_solutions\":" << stats.tactical_solutions
      << ",\"truncations\":" << stats.truncations
      << ",\"tree_nodes\":" << stats.tree_nodes
      << ",\"max_complete_turn_depth\":" << stats.max_complete_turn_depth
      << ",\"root_value\":" << stats.root_value << ",\"deadline_reached\":";
  write_bool(out, stats.deadline_reached);
  out << ",\"tree_cap_reached\":";
  write_bool(out, stats.tree_cap_reached);
  out << ",\"cached_continuation\":";
  write_bool(out, stats.cached_continuation);
  out << ",\"planned_action_length\":" << stats.planned_action_length
      << ",\"current_edge_index\":" << stats.current_edge_index
      << ",\"cached_moves_remaining\":" << stats.cached_moves_remaining
      << ",\"search_ordinal_in_game\":" << stats.searches << '}';
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
  out << ",\"rank5_derived\":";
  if (decision.rank5_derived_stats.has_value()) {
    write_rank5_derived_stats(out, *decision.rank5_derived_stats);
  } else {
    out << "null";
  }
  if (decision.deep_turn_search_stats.has_value()) {
    if (!decision.deep_turn_search_profile_nodes.has_value()) {
      throw std::logic_error(
          "DeepTurnSearch diagnostics are missing their profile node budget");
    }
    out << ",\"deep_turn_search\":";
    write_complete_turn_stats(out, *decision.deep_turn_search_stats,
                              *decision.deep_turn_search_profile_nodes);
  }
  out << ",\"jacek_replay_bfm\":";
  if (decision.jacek_replay_bfm_stats.has_value()) {
    if (!decision.jacek_replay_bfm_model_sha256.has_value()) {
      throw std::logic_error(
          "JacekReplayBfm diagnostics are missing the model SHA-256");
    }
    write_jacek_replay_bfm_stats(out, *decision.jacek_replay_bfm_stats,
                                 *decision.jacek_replay_bfm_model_sha256);
  } else {
    out << "null";
  }
  out << '}';
}

void write_opening(std::ostream &out, const OpeningReport &opening,
                   std::size_t requested_plies) {
  const std::size_t actual_plies = opening.transcript.size();
  out << "{\"pair_index\":" << opening.pair_index
      << ",\"generation_seed\":";
  write_uint64_string(out, opening.generation_seed);
  out << ",\"attempts\":" << opening.attempts;
  if (!opening.opening_id.empty()) {
    out << ",\"opening_id\":";
    write_string(out, opening.opening_id);
    out << ",\"phase\":";
    write_string(out, opening.phase);
    out << ",\"state_hash\":";
    write_string(out, opening.state_hash);
    out << ",\"canonical_key\":";
    write_string(out, opening.canonical_key);
  }
  out << ",\"requested_plies\":" << requested_plies
      << ",\"actual_plies\":" << actual_plies << ",\"moves\":[";
  for (std::size_t index = 0; index < opening.transcript.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    write_point(out, opening.transcript[index].to);
  }
  out << "],\"state\":{\"ball\":";
  write_point(out, opening.state.ball);
  out << ",\"to_move\":";
  write_string(out, player_name(opening.state.to_move));
  out << "}}";
}

void write_record(std::ostream &out, const Record &record) {
  const std::size_t scored_games = record.scored_games();
  const double percent =
      scored_games == 0 ? 0.0 : 100.0 * record.score() / scored_games;
  out << "{\"games\":" << record.games << ",\"wins\":" << record.wins
      << ",\"losses\":" << record.losses
      << ",\"truncations\":" << record.truncations
      << ",\"scored_games\":" << scored_games
      << ",\"score_points\":" << record.score()
      << ",\"score_percent\":" << percent << '}';
}

void write_timing(std::ostream &out, const TimingSummary &summary) {
  out << "{\"decisions\":" << summary.decisions
      << ",\"total_ns\":" << summary.total_ns
      << ",\"min_ns\":" << summary.min_ns
      << ",\"median_ns\":" << summary.median_ns
      << ",\"p90_ns\":" << summary.p90_ns
      << ",\"p95_ns\":" << summary.p95_ns
      << ",\"p99_ns\":" << summary.p99_ns
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
      << summary.budget_exhausted_searches
      << ",\"completed_turn_depth_zero_searches\":"
      << summary.completed_turn_depth_histogram[0]
      << ",\"completed_turn_depth_histogram\":{";
  bool first = true;
  for (std::size_t depth = 0;
       depth < summary.completed_turn_depth_histogram.size(); ++depth) {
    const std::size_t count = summary.completed_turn_depth_histogram[depth];
    if (count == 0) {
      continue;
    }
    if (!first) {
      out << ',';
    }
    first = false;
    write_string(out, std::to_string(depth));
    out << ':' << count;
  }
  out << "},\"attempted_turn_depth_histogram\":{";
  first = true;
  for (std::size_t depth = 0;
       depth < summary.attempted_turn_depth_histogram.size(); ++depth) {
    const std::size_t count = summary.attempted_turn_depth_histogram[depth];
    if (count == 0) {
      continue;
    }
    if (!first) {
      out << ',';
    }
    first = false;
    write_string(out, std::to_string(depth));
    out << ':' << count;
  }
  out << "}}";
}

void write_jacek_replay_bfm_summary(std::ostream &out,
                                    const JacekReplayBfmSummary &summary) {
  out << "{\"decisions\":" << summary.decisions
      << ",\"fresh_root_searches\":" << summary.fresh_root_searches
      << ",\"cached_continuation_edges\":" << summary.cached_continuation_edges
      << ",\"expansions_sum\":" << summary.expansions_sum
      << ",\"generated_actions_sum\":" << summary.generated_actions_sum
      << ",\"retained_actions_sum\":" << summary.retained_actions_sum
      << ",\"neural_evaluations_sum\":" << summary.neural_evaluations_sum
      << ",\"visits_sum\":" << summary.visits_sum
      << ",\"completed_actions_sum\":" << summary.completed_actions_sum
      << ",\"duplicate_boundaries_sum\":" << summary.duplicate_boundaries_sum
      << ",\"partial_paths_sum\":" << summary.partial_paths_sum
      << ",\"fifo_extractions_sum\":" << summary.fifo_extractions_sum
      << ",\"lifo_extractions_sum\":" << summary.lifo_extractions_sum
      << ",\"tactical_proofs_sum\":" << summary.tactical_proofs_sum
      << ",\"tactical_solutions_sum\":" << summary.tactical_solutions_sum
      << ",\"truncations_sum\":" << summary.truncations_sum
      << ",\"tree_nodes_sum\":" << summary.tree_nodes_sum
      << ",\"tree_nodes_max\":" << summary.tree_nodes_max
      << ",\"max_complete_turn_depth\":" << summary.max_complete_turn_depth
      << ",\"minimum_root_value\":";
  if (summary.minimum_root_value.has_value()) {
    out << *summary.minimum_root_value;
  } else {
    out << "null";
  }
  out << ",\"maximum_root_value\":";
  if (summary.maximum_root_value.has_value()) {
    out << *summary.maximum_root_value;
  } else {
    out << "null";
  }
  out << ",\"deadline_reached_searches\":" << summary.deadline_reached_searches
      << ",\"tree_cap_reached_searches\":" << summary.tree_cap_reached_searches
      << ",\"fresh_root_timing\":";
  write_timing(out, summary.fresh_root_timing);
  out << ",\"all_edge_timing\":";
  write_timing(out, summary.all_edge_timing);
  out << '}';
}

void write_complete_turn_summary(std::ostream &out,
                                 const Rank5DerivedSummary &summary,
                                 std::uint64_t requested_nodes) {
  out << "{\"decisions\":" << summary.decisions
      << ",\"fresh_root_searches\":" << summary.fresh_root_searches
      << ",\"cached_continuation_edges\":"
      << summary.cached_continuation_edges
      << ",\"requested_nodes_per_fresh_search\":"
      << requested_nodes
      << ",\"requested_nodes_sum\":" << summary.requested_nodes_sum
      << ",\"visited_nodes_sum\":" << summary.visited_nodes_sum
      << ",\"budget_exhausted_fresh_searches\":"
      << summary.budget_exhausted_fresh_searches
      << ",\"maximum_current_edge_index\":"
      << summary.maximum_current_edge_index
      << ",\"minimum_root_score\":";
  if (summary.minimum_root_score.has_value()) {
    out << *summary.minimum_root_score;
  } else {
    out << "null";
  }
  out << ",\"maximum_root_score\":";
  if (summary.maximum_root_score.has_value()) {
    out << *summary.maximum_root_score;
  } else {
    out << "null";
  }
  auto write_histogram = [&](const auto &histogram) {
    out << '{';
    bool first = true;
    for (const auto &[value, count] : histogram) {
      if (!first) {
        out << ',';
      }
      first = false;
      write_string(out, std::to_string(value));
      out << ':' << count;
    }
    out << '}';
  };
  out << ",\"completed_turn_depth_histogram\":";
  write_histogram(summary.completed_depth_histogram);
  out << ",\"attempted_turn_depth_histogram\":";
  write_histogram(summary.attempted_depth_histogram);
  out << ",\"planned_action_length_histogram\":";
  write_histogram(summary.planned_action_length_histogram);
  auto write_counter = [&](std::string_view name,
                           const Rank5DerivedCounterSummary &counter) {
    out << ",\"" << name << "_sum\":" << counter.sum << ",\"" << name
        << "_max\":" << counter.max;
  };
  write_counter("leaf_evaluations", summary.leaf_evaluations);
  write_counter("terminal_nodes", summary.terminal_nodes);
  write_counter("completed_actions", summary.completed_actions);
  write_counter("cutoffs", summary.cutoffs);
  write_counter("transposition_probes", summary.transposition_probes);
  write_counter("transposition_hits", summary.transposition_hits);
  write_counter("transposition_cutoffs", summary.transposition_cutoffs);
  write_counter("transposition_stores", summary.transposition_stores);
  write_counter("continuation_transposition_hits",
                summary.continuation_transposition_hits);
  write_counter("evaluation_cache_probes", summary.evaluation_cache_probes);
  write_counter("evaluation_cache_hits", summary.evaluation_cache_hits);
  write_counter("terminal_bound_cutoffs", summary.terminal_bound_cutoffs);
  write_counter("forced_edges", summary.forced_edges);
  write_counter("root_seed_actions", summary.root_seed_actions);
  write_counter("root_transposition_reuses",
                summary.root_transposition_reuses);
  write_counter("max_action_edges", summary.max_action_edges);
  out << ",\"fresh_root_timing\":";
  write_timing(out, summary.fresh_root_timing);
  out << ",\"all_edge_timing\":";
  write_timing(out, summary.all_edge_timing);
  out << '}';
}

std::optional<std::uint64_t> deep_profile_nodes(
    const std::vector<const DecisionReport *> &decisions) {
  std::optional<std::uint64_t> result;
  for (const DecisionReport *decision : decisions) {
    if (!decision->deep_turn_search_profile_nodes.has_value()) {
      continue;
    }
    if (result.has_value() &&
        *result != *decision->deep_turn_search_profile_nodes) {
      throw std::logic_error(
          "arena summary mixes DeepTurnSearch profile node budgets");
    }
    result = decision->deep_turn_search_profile_nodes;
  }
  return result;
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
  out << ",\"jacek_replay_bfm\":";
  write_jacek_replay_bfm_summary(out, summarize_jacek_replay_bfm(decisions));
  out << ",\"rank5_derived\":";
  write_complete_turn_summary(out, summarize_rank5_derived(decisions),
                              Rank5DerivedConfig::profile_max_nodes);
  if (const auto deep_nodes = deep_profile_nodes(decisions)) {
    out << ",\"deep_turn_search\":";
    write_complete_turn_summary(
        out, summarize_deep_turn_search(decisions, *deep_nodes), *deep_nodes);
  }
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
  out << ",\"jacek_replay_bfm\":";
  write_jacek_replay_bfm_summary(out, summarize_jacek_replay_bfm(decisions));
  out << ",\"rank5_derived\":";
  write_complete_turn_summary(out, summarize_rank5_derived(decisions),
                              Rank5DerivedConfig::profile_max_nodes);
  if (const auto deep_nodes = deep_profile_nodes(decisions)) {
    out << ",\"deep_turn_search\":";
    write_complete_turn_summary(
        out, summarize_deep_turn_search(decisions, *deep_nodes), *deep_nodes);
  }
  out << '}';
}

}  // namespace detail

std::string build_provenance_json() {
  using namespace detail;
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << "{\"schema\":\"papersoccer.arena-build.v1\",\"runtime\":";
#ifdef __EMSCRIPTEN__
  write_string(out, "wasm");
#else
  write_string(out, "native");
#endif
  out << ",\"build_type\":";
  write_string(out, PAPERSOCCER_BUILD_TYPE);
  out << ",\"ndebug\":";
#ifdef NDEBUG
  write_bool(out, true);
#else
  write_bool(out, false);
#endif
  out << ",\"sanitizers_enabled\":";
  write_bool(out, PAPERSOCCER_SANITIZERS_ENABLED);
  out << ",\"compiler_id\":";
  write_string(out, PAPERSOCCER_COMPILER_ID);
  out << ",\"compiler_version\":";
  write_string(out, PAPERSOCCER_COMPILER_VERSION);
  out << ",\"configured_flags\":";
  write_string(out, PAPERSOCCER_CONFIGURED_FLAGS);
  out << ",\"cxx_standard\":" << __cplusplus << ",\"source_commit\":";
  write_string(out, PAPERSOCCER_SOURCE_COMMIT);
  out << ",\"source_dirty\":";
  write_bool(out, PAPERSOCCER_SOURCE_DIRTY);
  out << '}';
  return out.str();
}

std::string run_matches_json(const MatchesConfig &config) {
  using namespace detail;
  validate_common(config.rules, config.candidate, config.reference);
  const bool frozen_opening_mode = !config.frozen_openings.empty();
  if (config.seed_pairs == 0) {
    throw std::invalid_argument("arena seed pair count must be greater than zero");
  }
  if (config.max_plies == 0) {
    throw std::invalid_argument("arena max plies must be greater than zero");
  }
  if (config.bootstrap_samples == 0) {
    throw std::invalid_argument("arena bootstrap samples must be greater than zero");
  }
  if (config.seed_pairs > std::numeric_limits<std::size_t>::max() / 2) {
    throw std::invalid_argument("arena seed pair count is too large");
  }

  std::vector<OpeningReport> openings;
  std::size_t effective_opening_plies = config.opening_plies;
  if (frozen_opening_mode) {
    if (config.opening_plies != 0) {
      throw std::invalid_argument(
          "arena frozen openings are incompatible with generated opening plies");
    }
    if (config.frozen_openings.size() != config.seed_pairs) {
      throw std::invalid_argument(
          "arena frozen opening count must equal the seed pair count");
    }
    openings = validate_frozen_openings(config.rules, config.frozen_openings);
    effective_opening_plies = openings.front().transcript.size();
  }
  if (effective_opening_plies >= config.max_plies &&
      effective_opening_plies != 0) {
    throw std::invalid_argument(
        "arena opening plies must be less than max plies");
  }

  warm_up_match_entrants(config.rules, config.candidate, config.reference,
                         config.base_seed, config.warmup_decisions);

  SplitMix64 seeds{config.base_seed};
  SplitMix64 opening_pair_seeds{config.base_seed ^ kOpeningSeedSalt};
  if (!frozen_opening_mode && effective_opening_plies != 0) {
    openings.reserve(config.seed_pairs);
  }
  std::vector<GameReport> games;
  games.reserve(config.seed_pairs * 2);
  for (std::size_t pair = 0; pair < config.seed_pairs; ++pair) {
    const std::uint64_t candidate_seed = seeds.next();
    const std::uint64_t reference_seed = seeds.next();
    GameState initial_state = make_initial_state(config.rules);
    if (frozen_opening_mode) {
      initial_state = openings[pair].state;
    } else if (effective_opening_plies != 0) {
      OpeningReport opening = generate_opening(
          pair, config.rules, opening_pair_seeds.next(),
          effective_opening_plies);
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

  const std::vector<std::optional<double>> pair_scores =
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
      << "},\"rules_profile\":";
  write_string(out, rules_profile_name(config.rules));
  out << ",\"base_seed\":";
  write_uint64_string(out, config.base_seed);
  out << ",\"seed_derivation\":\"splitmix64\""
      << ",\"seed_pairs\":" << config.seed_pairs
      << ",\"games\":" << games.size()
      << ",\"opening_plies\":" << effective_opening_plies
      << ",\"opening_generator\":";
  write_string(out, frozen_opening_mode
                        ? "frozen_uniform_legal_move_data_generation_bank"
                        : "uniform_random");
  out << ",\"opening_seed_derivation\":";
  write_string(out, frozen_opening_mode
                        ? "committed_bank_accepted_generation_seeds"
                        : "domain_separated_splitmix64");
  out << ",\"max_plies\":" << config.max_plies
      << ",\"bootstrap_samples\":" << config.bootstrap_samples
      << ",\"warmup\":{\"decisions_per_entrant\":"
      << config.warmup_decisions
      << ",\"timed\":false,\"generation_plies\":"
      << kWarmupGenerationPlies
      << ",\"position_generator\":"
         "\"uniform_legal_move_generator\",\"seed_derivation\":"
         "\"domain_separated_splitmix64\",\"bot_instances\":"
         "\"separate_from_measured_games\"}"
      << ",\"candidate\":";
  write_bot_config(out, config.candidate);
  out << ",\"reference\":";
  write_bot_config(out, config.reference);
  out << "},\"openings\":[";
  for (std::size_t index = 0; index < openings.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    write_opening(out, openings[index], effective_opening_plies);
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
    write_optional_double(out, pair_scores[index]);
  }
  out << "],\"pair_bootstrap_95\":{\"method\":"
         "\"deterministic_pair_resampling\",\"seed\":";
  write_uint64_string(out, interval.seed);
  out << ",\"samples\":" << interval.samples
      << ",\"confidence\":0.95,\"valid_pairs\":"
      << interval.valid_pairs << ",\"invalid_pairs\":"
      << interval.invalid_pairs << ",\"lower_percent\":";
  write_optional_double(out, interval.lower_percent);
  out << ",\"upper_percent\":";
  write_optional_double(out, interval.upper_percent);
  out << "}}}";
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
  const std::size_t move_agreements = static_cast<std::size_t>(std::count_if(
      positions.begin(), positions.end(), [](const PositionReport &position) {
        return position.candidate.move == position.reference.move;
      }));
  const std::size_t move_changes = positions.size() - move_agreements;

  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(17);
  out << "{\"schema\":\"papersoccer.arena.v1\",\"mode\":\"positions\","
         "\"runtime\":";
  write_string(out, runtime_name());
  out << ",\"timing_unit\":\"nanoseconds\","
         "\"configuration\":{\"rules\":{\"width\":"
      << config.rules.width << ",\"height\":" << config.rules.height
      << "},\"rules_profile\":";
  write_string(out, rules_profile_name(config.rules));
  out << ",\"base_seed\":";
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
      out << ",\"rank5_derived\":";
      if (evaluation.rank5_derived_stats.has_value()) {
        write_rank5_derived_stats(out, *evaluation.rank5_derived_stats);
      } else {
        out << "null";
      }
      if (evaluation.deep_turn_search_stats.has_value()) {
        if (!evaluation.deep_turn_search_profile_nodes.has_value()) {
          throw std::logic_error(
              "DeepTurnSearch evaluation is missing its profile node budget");
        }
        out << ",\"deep_turn_search\":";
        write_complete_turn_stats(out, *evaluation.deep_turn_search_stats,
                                  *evaluation.deep_turn_search_profile_nodes);
      }
      out << ",\"jacek_replay_bfm\":";
      if (evaluation.jacek_replay_bfm_stats.has_value()) {
        if (!evaluation.jacek_replay_bfm_model_sha256.has_value()) {
          throw std::logic_error(
              "JacekReplayBfm evaluation is missing the model SHA-256");
        }
        write_jacek_replay_bfm_stats(out, *evaluation.jacek_replay_bfm_stats,
                                     *evaluation.jacek_replay_bfm_model_sha256);
      } else {
        out << "null";
      }
      out << '}';
    }
    out << "]}";
  }
  out << "],\"summary\":{\"illegal_moves\":0,\"move_comparison\":{"
      << "\"positions\":" << positions.size()
      << ",\"agreements\":" << move_agreements
      << ",\"changes\":" << move_changes << "},\"candidate\":";
  write_position_entrant_summary(out, Entrant::Candidate, positions);
  out << ",\"reference\":";
  write_position_entrant_summary(out, Entrant::Reference, positions);
  out << "}}";
  return out.str();
}

}  // namespace papersoccer::arena
