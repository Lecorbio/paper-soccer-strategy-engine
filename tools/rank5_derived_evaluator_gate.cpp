#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/game_review.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

constexpr std::uint64_t kDefaultNodes = 50'000;
constexpr std::uint32_t kDefaultDepth = 32;
constexpr std::size_t kDefaultTranspositionEntries = 65'536;
constexpr std::size_t kDefaultEvaluationEntries = 32'768;
constexpr int kDefaultPairsPerStage = 20;
constexpr int kDefaultBootstrapSamples = 10'000;
constexpr int kDefaultMaximumPhysicalPlies = 1'024;
constexpr std::uint64_t kOpeningSeedBase = 0x52414e4b35474154ULL;
constexpr std::uint64_t kBootstrapSeed = 0x4253545250354349ULL;
constexpr std::array<int, 3> kOpeningPhysicalPlies{4, 12, 20};

struct Options {
  std::uint64_t nodes{kDefaultNodes};
  std::uint32_t depth{kDefaultDepth};
  int pairs_per_stage{kDefaultPairsPerStage};
  int bootstrap_samples{kDefaultBootstrapSamples};
  int maximum_physical_plies{kDefaultMaximumPhysicalPlies};
  std::string output;
  std::string summary;
};

struct Opening {
  int stage_physical_plies{};
  int index{};
  std::uint64_t requested_seed{};
  std::uint64_t effective_seed{};
  std::uint64_t attempts{};
  ps::GameState state{};
  std::vector<ps::Move> sequence{};
};

struct OperationalCounts {
  std::uint64_t illegal_moves{};
  std::uint64_t errors{};
  std::uint64_t unexplained_truncations{};
  std::uint64_t incomplete_actions{};
  std::uint64_t candidate_searches{};
  std::uint64_t reference_searches{};
  std::uint64_t candidate_cached_edges{};
  std::uint64_t reference_cached_edges{};
  std::uint64_t candidate_nodes{};
  std::uint64_t reference_nodes{};
  std::uint64_t candidate_budget_exhaustions{};
  std::uint64_t reference_budget_exhaustions{};
};

struct GameRecord {
  int candidate_player{};
  std::optional<int> winner{};
  std::optional<double> candidate_score{};
  int physical_plies{};
  bool complete{};
  bool truncated{};
  std::string error{};
  OperationalCounts operations{};
};

struct PairRecord {
  Opening opening{};
  std::array<GameRecord, 2> games{};
  std::optional<double> candidate_score{};
};

struct Interval {
  double estimate{};
  double lower{};
  double upper{};
};

struct OutcomeSummary {
  std::size_t wins{};
  std::size_t losses{};
  std::vector<double> pair_scores{};
};

std::uint64_t next_random(std::uint64_t &state) noexcept {
  state ^= state >> 12U;
  state ^= state << 25U;
  state ^= state >> 27U;
  return state * 0x2545f4914f6cdd1dULL;
}

std::string json_string(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (character < 0x20U) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(character) << std::dec;
        } else {
          out << character;
        }
    }
  }
  out << '"';
  return out.str();
}

std::string hex64(std::uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

std::uint64_t fnv1a_update(std::uint64_t hash, std::string_view bytes) {
  for (const unsigned char byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string sequence_text(const Opening &opening) {
  std::ostringstream out;
  ps::Point from{4, 6};
  for (std::size_t index = 0; index < opening.sequence.size(); ++index) {
    if (index != 0) {
      out << ';';
    }
    const ps::Point to = opening.sequence[index].to;
    out << from.x << ',' << from.y << '-' << to.x << ',' << to.y;
    from = to;
  }
  return out.str();
}

bool contains(const std::vector<ps::Move> &moves, ps::Move candidate) {
  return std::find(moves.begin(), moves.end(), candidate) != moves.end();
}

ps::CompleteTurnAnalysisConfig analysis_config(const Options &options) {
  ps::CompleteTurnAnalysisConfig config;
  config.max_turn_depth = options.depth;
  config.max_nodes = options.nodes;
  config.transposition_table_entries = kDefaultTranspositionEntries;
  config.evaluation_table_entries = kDefaultEvaluationEntries;
  return config;
}

bool same_state(const ps::GameState &left, const ps::GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

// This intentionally is not a Rank5DerivedBot. It keeps the historical gate
// executable useful for small deterministic regression runs while making
// configurable work report under the public complete-turn analysis identity.
class ConfiguredAnalysisBot {
 public:
  explicit ConfiguredAnalysisBot(ps::CompleteTurnAnalysisConfig config)
      : analyzer_(config) {}

  ps::Move choose_move(const ps::GameState &state) {
    if (expected_state_.has_value() && same_state(state, *expected_state_) &&
        next_ < action_.size()) {
      const ps::Move move = action_[next_++];
      stats_.nodes = 0;
      stats_.cached_continuation = true;
      stats_.planned_action_length = action_.size();
      stats_.current_edge_index = next_ - 1U;
      stats_.cached_moves_remaining = action_.size() - next_;
      if (next_ < action_.size()) {
        expected_state_ = ps::apply_move(state, move);
      } else {
        clear_cache();
      }
      return move;
    }

    clear_cache();
    ps::CompleteTurnAnalysis analysis = analyzer_.analyze(state);
    action_ = std::move(analysis.action);
    stats_ = std::move(analysis.stats);
    ++searches_;
    stats_.planned_action_length = action_.size();
    stats_.current_edge_index = 0;
    stats_.cached_moves_remaining = action_.size() - 1U;
    stats_.searches = searches_;
    const ps::Move move = action_.front();
    next_ = 1;
    if (next_ < action_.size()) {
      expected_state_ = ps::apply_move(state, move);
    } else {
      clear_cache();
    }
    return move;
  }

  const ps::CompleteTurnSearchStats &last_search_stats() const noexcept {
    return stats_;
  }

 private:
  ps::CompleteTurnAnalyzer analyzer_;
  ps::CompleteTurnSearchStats stats_{};
  std::vector<ps::Move> action_{};
  std::size_t next_{};
  std::optional<ps::GameState> expected_state_{};
  std::uint64_t searches_{};

  void clear_cache() noexcept {
    action_.clear();
    next_ = 0;
    expected_state_.reset();
    stats_.cached_moves_remaining = 0;
  }
};

Opening make_opening(int stage, int index) {
  const std::uint64_t requested_seed =
      kOpeningSeedBase ^ (static_cast<std::uint64_t>(stage) << 40U) ^
      (static_cast<std::uint64_t>(index + 1) * 0x9e3779b97f4a7c15ULL);
  for (std::uint64_t attempt = 0; attempt < 100'000; ++attempt) {
    const std::uint64_t effective_seed =
        requested_seed + attempt * 0xd1b54a32d192ed03ULL;
    std::uint64_t random = effective_seed == 0 ? 1 : effective_seed;
    ps::GameState state = ps::make_initial_state();
    std::vector<ps::Move> sequence;
    sequence.reserve(static_cast<std::size_t>(stage));
    for (int ply = 0; ply < stage && !ps::is_terminal(state); ++ply) {
      const std::vector<ps::Move> legal = ps::legal_moves(state);
      if (legal.empty()) {
        break;
      }
      const ps::Move move = legal[next_random(random) % legal.size()];
      sequence.push_back(move);
      state = ps::apply_move(state, move);
    }
    if (static_cast<int>(sequence.size()) == stage &&
        !ps::is_terminal(state)) {
      return Opening{stage, index, requested_seed, effective_seed, attempt,
                     std::move(state), std::move(sequence)};
    }
  }
  throw std::runtime_error("could not generate a non-terminal frozen opening");
}

void add_counts(OperationalCounts &target, const OperationalCounts &source) {
  target.illegal_moves += source.illegal_moves;
  target.errors += source.errors;
  target.unexplained_truncations += source.unexplained_truncations;
  target.incomplete_actions += source.incomplete_actions;
  target.candidate_searches += source.candidate_searches;
  target.reference_searches += source.reference_searches;
  target.candidate_cached_edges += source.candidate_cached_edges;
  target.reference_cached_edges += source.reference_cached_edges;
  target.candidate_nodes += source.candidate_nodes;
  target.reference_nodes += source.reference_nodes;
  target.candidate_budget_exhaustions += source.candidate_budget_exhaustions;
  target.reference_budget_exhaustions += source.reference_budget_exhaustions;
}

GameRecord play_game(const Opening &opening, int candidate_player,
                     const Options &options) {
  GameRecord record;
  record.candidate_player = candidate_player;
  ps::GameState state = opening.state;
  ConfiguredAnalysisBot candidate(analysis_config(options));
  ps::Rank5DerivedBot reference;
  std::array<std::size_t, 2> continuation_edges{};
  std::array<std::size_t, 2> planned_action_lengths{};

  try {
    while (!ps::is_terminal(state) &&
           record.physical_plies < options.maximum_physical_plies) {
      const int mover = state.to_move == ps::Player::One ? 0 : 1;
      const bool is_candidate = mover == candidate_player;
      const std::vector<ps::Move> legal = ps::legal_moves(state);
      if (legal.empty()) {
        throw std::logic_error("in-progress state has no legal moves");
      }
      const ps::Move move = is_candidate ? candidate.choose_move(state)
                                         : reference.choose_move(state);
      const ps::CompleteTurnSearchStats stats =
          is_candidate ? candidate.last_search_stats()
                       : reference.last_search_stats();
      if (!contains(legal, move)) {
        ++record.operations.illegal_moves;
        throw std::logic_error("bot returned an illegal move");
      }

      if (stats.cached_continuation) {
        if (continuation_edges[mover] == 0 || stats.nodes != 0 ||
            stats.planned_action_length != planned_action_lengths[mover] ||
            stats.current_edge_index + 1U !=
                stats.planned_action_length - continuation_edges[mover] + 1U) {
          ++record.operations.incomplete_actions;
          throw std::logic_error("cached continuation metadata is inconsistent");
        }
        --continuation_edges[mover];
        if (is_candidate) {
          ++record.operations.candidate_cached_edges;
        } else {
          ++record.operations.reference_cached_edges;
        }
      } else {
        if (continuation_edges[mover] != 0 ||
            stats.planned_action_length == 0 ||
            stats.current_edge_index != 0) {
          ++record.operations.incomplete_actions;
          throw std::logic_error("fresh search did not start a complete action");
        }
        planned_action_lengths[mover] = stats.planned_action_length;
        continuation_edges[mover] = stats.planned_action_length - 1U;
        if (is_candidate) {
          ++record.operations.candidate_searches;
          record.operations.candidate_nodes += stats.nodes;
          record.operations.candidate_budget_exhaustions +=
              stats.budget_exhausted ? 1U : 0U;
        } else {
          ++record.operations.reference_searches;
          record.operations.reference_nodes += stats.nodes;
          record.operations.reference_budget_exhaustions +=
              stats.budget_exhausted ? 1U : 0U;
        }
      }

      const ps::Player before_player = state.to_move;
      state = ps::apply_move(state, move);
      ++record.physical_plies;
      if (continuation_edges[mover] == 0) {
        if (!ps::is_terminal(state) && state.to_move == before_player) {
          ++record.operations.incomplete_actions;
          throw std::logic_error("complete action ended before the turn changed");
        }
      } else if (ps::is_terminal(state) || state.to_move != before_player) {
        ++record.operations.incomplete_actions;
        throw std::logic_error("planned action extended past the turn boundary");
      }
    }
  } catch (const std::exception &error) {
    ++record.operations.errors;
    record.error = error.what();
  } catch (...) {
    ++record.operations.errors;
    record.error = "unknown exception";
  }

  if (!ps::is_terminal(state) && record.error.empty()) {
    record.truncated = true;
    ++record.operations.unexplained_truncations;
  }
  if (ps::is_terminal(state) && record.error.empty()) {
    record.complete = true;
    const std::optional<ps::Player> winning_player = ps::winner(state);
    if (!winning_player.has_value()) {
      ++record.operations.errors;
      record.error = "terminal state has no winner";
      record.complete = false;
    } else {
      record.winner = *winning_player == ps::Player::One ? 0 : 1;
      record.candidate_score = *record.winner == candidate_player ? 1.0 : 0.0;
    }
  }
  return record;
}

Interval bootstrap_interval(const std::vector<double> &scores, int samples,
                            std::uint64_t seed) {
  if (scores.empty()) {
    return {};
  }
  double total = 0.0;
  for (const double score : scores) {
    total += score;
  }
  std::vector<double> estimates;
  estimates.reserve(static_cast<std::size_t>(samples));
  std::uint64_t random = seed == 0 ? 1 : seed;
  for (int sample = 0; sample < samples; ++sample) {
    double sum = 0.0;
    for (std::size_t draw = 0; draw < scores.size(); ++draw) {
      sum += scores[next_random(random) % scores.size()];
    }
    estimates.push_back(sum / static_cast<double>(scores.size()));
  }
  std::sort(estimates.begin(), estimates.end());
  const auto quantile = [&](double fraction) {
    const std::size_t index = static_cast<std::size_t>(
        fraction * static_cast<double>(estimates.size() - 1U));
    return estimates[index];
  };
  return {total / static_cast<double>(scores.size()), quantile(0.025),
          quantile(0.975)};
}

void write_counts(std::ostream &out, const OperationalCounts &counts) {
  out << "{\"illegal_moves\":" << counts.illegal_moves
      << ",\"errors\":" << counts.errors
      << ",\"unexplained_truncations\":" << counts.unexplained_truncations
      << ",\"incomplete_actions\":" << counts.incomplete_actions
      << ",\"candidate_searches\":" << counts.candidate_searches
      << ",\"reference_searches\":" << counts.reference_searches
      << ",\"candidate_cached_edges\":" << counts.candidate_cached_edges
      << ",\"reference_cached_edges\":" << counts.reference_cached_edges
      << ",\"candidate_nodes\":" << counts.candidate_nodes
      << ",\"reference_nodes\":" << counts.reference_nodes
      << ",\"candidate_budget_exhaustions\":"
      << counts.candidate_budget_exhaustions
      << ",\"reference_budget_exhaustions\":"
      << counts.reference_budget_exhaustions << '}';
}

void write_game(std::ostream &out, const GameRecord &game) {
  out << "{\"candidate_player\":" << game.candidate_player << ",\"winner\":";
  if (game.winner.has_value()) out << *game.winner; else out << "null";
  out << ",\"candidate_score\":";
  if (game.candidate_score.has_value()) out << *game.candidate_score;
  else out << "null";
  out << ",\"physical_plies\":" << game.physical_plies
      << ",\"complete\":" << (game.complete ? "true" : "false")
      << ",\"truncated\":" << (game.truncated ? "true" : "false")
      << ",\"error\":" << json_string(game.error) << ",\"operations\":";
  write_counts(out, game.operations);
  out << '}';
}

void write_interval(std::ostream &out, const Interval &interval,
                    std::size_t records, int samples) {
  out << "{\"unit\":\"color_swapped_pair\",\"method\":"
         "\"paired_percentile_bootstrap\",\"confidence\":0.95,"
         "\"resamples\":"
      << samples << ",\"records\":" << records << ",\"estimate\":"
      << interval.estimate << ",\"lower\":" << interval.lower
      << ",\"upper\":" << interval.upper << '}';
}

OutcomeSummary summarize_outcomes(const std::vector<PairRecord> &pairs) {
  OutcomeSummary summary;
  for (const PairRecord &pair : pairs) {
    for (const GameRecord &game : pair.games) {
      if (game.candidate_score == 1.0) {
        ++summary.wins;
      } else if (game.candidate_score == 0.0) {
        ++summary.losses;
      }
    }
    if (pair.candidate_score.has_value()) {
      summary.pair_scores.push_back(*pair.candidate_score);
    }
  }
  return summary;
}

long long parse_positive(std::string_view raw, std::string_view label) {
  std::string text(raw);
  std::size_t consumed = 0;
  const long long value = std::stoll(text, &consumed, 10);
  if (consumed != text.size() || value <= 0) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      std::cout << "usage: complete_turn_analysis_regression [--nodes N] "
                   "[--depth N] [--pairs-per-stage N] [--bootstrap N] "
                   "[--max-plies N] [--output RAW.json] "
                   "[--summary SUMMARY.json]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--nodes") {
      options.nodes = static_cast<std::uint64_t>(parse_positive(value, "nodes"));
    } else if (argument == "--depth") {
      options.depth = static_cast<std::uint32_t>(parse_positive(value, "depth"));
    } else if (argument == "--pairs-per-stage") {
      options.pairs_per_stage = static_cast<int>(parse_positive(value, "pairs"));
    } else if (argument == "--bootstrap") {
      options.bootstrap_samples = static_cast<int>(parse_positive(value, "bootstrap"));
    } else if (argument == "--max-plies") {
      options.maximum_physical_plies = static_cast<int>(parse_positive(value, "max plies"));
    } else if (argument == "--output") {
      options.output = value;
    } else if (argument == "--summary") {
      options.summary = value;
    } else {
      throw std::invalid_argument("unknown argument: " + std::string(argument));
    }
  }
  if (options.depth > ps::Rank5DerivedConfig::maximum_turn_depth) {
    throw std::invalid_argument("depth exceeds complete-turn analysis maximum");
  }
  return options;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<std::vector<PairRecord>> stages;
    stages.reserve(kOpeningPhysicalPlies.size());
    OperationalCounts overall_operations;
    std::vector<double> overall_scores;
    std::uint64_t opening_fingerprint = 14695981039346656037ULL;

    for (const int stage : kOpeningPhysicalPlies) {
      std::cerr << "complete-turn regression: stage " << stage
                << " physical plies\n";
      std::vector<PairRecord> pairs;
      pairs.reserve(static_cast<std::size_t>(options.pairs_per_stage));
      for (int index = 0; index < options.pairs_per_stage; ++index) {
        Opening opening = make_opening(stage, index);
        const std::string sequence = sequence_text(opening);
        opening_fingerprint = fnv1a_update(opening_fingerprint, sequence);
        opening_fingerprint = fnv1a_update(opening_fingerprint, "\n");
        PairRecord pair;
        pair.opening = opening;
        pair.games[0] = play_game(opening, 0, options);
        pair.games[1] = play_game(opening, 1, options);
        add_counts(overall_operations, pair.games[0].operations);
        add_counts(overall_operations, pair.games[1].operations);
        if (pair.games[0].candidate_score.has_value() &&
            pair.games[1].candidate_score.has_value()) {
          pair.candidate_score =
              (*pair.games[0].candidate_score + *pair.games[1].candidate_score) /
              2.0;
          overall_scores.push_back(*pair.candidate_score);
        }
        pairs.push_back(std::move(pair));
        std::cerr << "complete-turn regression: stage=" << stage
                  << " pair=" << index + 1
                  << '/' << options.pairs_per_stage << '\n';
      }
      stages.push_back(std::move(pairs));
    }

    const Interval overall_ci = bootstrap_interval(
        overall_scores, options.bootstrap_samples, kBootstrapSeed);
    const std::size_t expected_pairs = kOpeningPhysicalPlies.size() *
                                       static_cast<std::size_t>(options.pairs_per_stage);
    const bool operationally_clean = overall_operations.illegal_moves == 0 &&
        overall_operations.errors == 0 &&
        overall_operations.unexplained_truncations == 0 &&
        overall_operations.incomplete_actions == 0 &&
        overall_scores.size() == expected_pairs;
    const bool candidate_selected =
        operationally_clean && overall_ci.lower > 0.5;
    const ps::CompleteTurnAnalysisConfig candidate_config =
        analysis_config(options);

    std::ostringstream json;
    json << std::setprecision(17);
    json << "{\n  \"schema\": \"papersoccer.complete-turn-regression.v1\","
            "\n  \"config\": {\"rules\":{\"width\":8,\"height\":10,"
            "\"goal_rule\":\"opponent_goal_only\","
            "\"blocked_rule\":\"player_to_move_loses\"},"
            "\"candidate_identity\":\"complete-turn-analysis\","
            "\"candidate_profile\":"
         << json_string(candidate_config.profile_name())
         << ",\"reference_identity\":\"rank5-derived-fixed-50k\","
            "\"learned_value_blend_percent\":0,\"max_nodes\":"
         << options.nodes
         << ",\"max_turn_depth\":" << options.depth
         << ",\"transposition_entries\":" << kDefaultTranspositionEntries
         << ",\"evaluation_entries\":" << kDefaultEvaluationEntries
         << ",\"max_time_ms\":0,\"replay_corrections\":false,"
            "\"opening_book\":false,\"pairs_per_stage\":"
         << options.pairs_per_stage
         << ",\"opening_physical_plies\":[4,12,20],"
            "\"maximum_game_physical_plies\":" << options.maximum_physical_plies
         << ",\"bootstrap\":{\"method\":\"paired_percentile\","
            "\"resamples\":" << options.bootstrap_samples
         << ",\"seed\":\"0x" << hex64(kBootstrapSeed)
         << "\",\"confidence\":0.95},\"selection_rule\":"
            "\"select configurable analysis only when all operational counts "
            "are zero and the overall paired-bootstrap lower bound is strictly "
            "greater than 0.50; otherwise retain fixed Rank5Derived\"},\n"
         << "  \"openings\": {\"generator\":\"xorshift64star/legal-index/v1\","
            "\"seed_base\":\"0x" << hex64(kOpeningSeedBase)
         << "\",\"fingerprint\":{\"algorithm\":\"fnv1a64\",\"value\":\""
         << hex64(opening_fingerprint) << "\"},\"records\":[";
    bool first_opening = true;
    for (const auto &stage : stages) {
      for (const PairRecord &pair : stage) {
        if (!first_opening) json << ',';
        first_opening = false;
        json << "{\"stage_physical_plies\":" << pair.opening.stage_physical_plies
             << ",\"index\":" << pair.opening.index
             << ",\"requested_seed\":\"0x" << hex64(pair.opening.requested_seed)
             << "\",\"effective_seed\":\"0x" << hex64(pair.opening.effective_seed)
             << "\",\"rejected_attempts\":" << pair.opening.attempts
             << ",\"sequence\":" << json_string(sequence_text(pair.opening))
             << '}';
      }
    }
    json << "]},\n  \"stages\": [";
    for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
      if (stage_index != 0) json << ',';
      const auto &stage = stages[stage_index];
      std::vector<double> scores;
      OperationalCounts operations;
      for (const PairRecord &pair : stage) {
        add_counts(operations, pair.games[0].operations);
        add_counts(operations, pair.games[1].operations);
        if (pair.candidate_score.has_value()) scores.push_back(*pair.candidate_score);
      }
      const Interval interval = bootstrap_interval(
          scores, options.bootstrap_samples,
          kBootstrapSeed ^ static_cast<std::uint64_t>(kOpeningPhysicalPlies[stage_index]));
      json << "{\"opening_physical_plies\":"
           << kOpeningPhysicalPlies[stage_index] << ",\"records\":[";
      for (std::size_t pair_index = 0; pair_index < stage.size(); ++pair_index) {
        if (pair_index != 0) json << ',';
        const PairRecord &pair = stage[pair_index];
        json << "{\"opening_index\":" << pair.opening.index
             << ",\"games\":[";
        write_game(json, pair.games[0]);
        json << ',';
        write_game(json, pair.games[1]);
        json << "],\"candidate_pair_score\":";
        if (pair.candidate_score.has_value()) json << *pair.candidate_score;
        else json << "null";
        json << '}';
      }
      json << "],\"operational_counts\":";
      write_counts(json, operations);
      json << ",\"confidence_interval\":";
      write_interval(json, interval, scores.size(), options.bootstrap_samples);
      json << '}';
    }
    json << "],\n  \"overall\": {\"records\":[";
    bool first_overall_record = true;
    for (const auto &stage : stages) {
      for (const PairRecord &pair : stage) {
        if (!first_overall_record) json << ',';
        first_overall_record = false;
        json << "{\"opening_physical_plies\":"
             << pair.opening.stage_physical_plies
             << ",\"opening_index\":" << pair.opening.index
             << ",\"candidate_pair_score\":";
        if (pair.candidate_score.has_value()) json << *pair.candidate_score;
        else json << "null";
        json << '}';
      }
    }
    json << "],\"expected_pair_records\":" << expected_pairs
         << ",\"complete_pair_records\":" << overall_scores.size()
         << ",\"operational_counts\":";
    write_counts(json, overall_operations);
    json << ",\"confidence_interval\":";
    write_interval(json, overall_ci, overall_scores.size(), options.bootstrap_samples);
    json << "},\n  \"selection\": {\"operationally_clean\":"
         << (operationally_clean ? "true" : "false")
         << ",\"lower_bound_strictly_above_half\":"
         << (overall_ci.lower > 0.5 ? "true" : "false")
         << ",\"selected_identity\":"
         << json_string(candidate_selected ? "complete-turn-analysis"
                                           : "rank5-derived-fixed-50k")
         << "}\n}\n";

    std::ostringstream summary;
    summary << std::setprecision(17);
    summary << "{\n  \"schema\": "
               "\"papersoccer.complete-turn-regression-summary.v1\","
               "\n  \"config\": {\"candidate_identity\":"
               "\"complete-turn-analysis\",\"candidate_profile\":"
            << json_string(candidate_config.profile_name())
            << ",\"reference_identity\":\"rank5-derived-fixed-50k\","
               "\"learned_value_blend_percent\":0,\"max_nodes\":"
            << options.nodes
            << ",\"max_turn_depth\":" << options.depth
            << ",\"pairs_per_stage\":" << options.pairs_per_stage
            << ",\"bootstrap_resamples\":" << options.bootstrap_samples
            << "},\n  \"opening_fingerprint\": {\"algorithm\":\"fnv1a64\","
               "\"value\":\"" << hex64(opening_fingerprint)
            << "\"},\n  \"stages\": [";
    for (std::size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
      if (stage_index != 0) summary << ',';
      const OutcomeSummary outcomes = summarize_outcomes(stages[stage_index]);
      const Interval interval = bootstrap_interval(
          outcomes.pair_scores, options.bootstrap_samples,
          kBootstrapSeed ^ static_cast<std::uint64_t>(
              kOpeningPhysicalPlies[stage_index]));
      summary << "{\"opening_physical_plies\":"
              << kOpeningPhysicalPlies[stage_index]
              << ",\"candidate_wins\":" << outcomes.wins
              << ",\"candidate_losses\":" << outcomes.losses
              << ",\"complete_pairs\":" << outcomes.pair_scores.size()
              << ",\"candidate_paired_score\":" << interval.estimate
              << ",\"ci_lower\":" << interval.lower
              << ",\"ci_upper\":" << interval.upper << '}';
    }
    std::size_t overall_wins = 0;
    std::size_t overall_losses = 0;
    for (const auto &stage : stages) {
      const OutcomeSummary outcomes = summarize_outcomes(stage);
      overall_wins += outcomes.wins;
      overall_losses += outcomes.losses;
    }
    summary << "],\n  \"overall\": {\"candidate_wins\":" << overall_wins
            << ",\"candidate_losses\":" << overall_losses
            << ",\"complete_pairs\":" << overall_scores.size()
            << ",\"candidate_paired_score\":" << overall_ci.estimate
            << ",\"ci_lower\":" << overall_ci.lower
            << ",\"ci_upper\":" << overall_ci.upper
            << ",\"operational_counts\":";
    write_counts(summary, overall_operations);
    summary << "},\n  \"selection\": {\"operationally_clean\":"
            << (operationally_clean ? "true" : "false")
            << ",\"lower_bound_strictly_above_half\":"
            << (overall_ci.lower > 0.5 ? "true" : "false")
            << ",\"selected_identity\":"
            << json_string(candidate_selected ? "complete-turn-analysis"
                                              : "rank5-derived-fixed-50k")
            << "}\n}\n";

    if (options.output.empty()) {
      std::cout << json.str();
    } else {
      std::ofstream output(options.output);
      if (!output) throw std::runtime_error("could not open output file");
      output << json.str();
      if (!output) throw std::runtime_error("could not write output file");
      std::cerr << "complete-turn regression: wrote " << options.output
                << '\n';
    }
    if (!options.summary.empty()) {
      std::ofstream output(options.summary);
      if (!output) throw std::runtime_error("could not open summary file");
      output << summary.str();
      if (!output) throw std::runtime_error("could not write summary file");
      std::cerr << "complete-turn regression: wrote " << options.summary
                << '\n';
    }
    return operationally_clean ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "complete-turn regression: " << error.what() << '\n';
    return 64;
  }
}
