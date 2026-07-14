#include "papersoccer/arena.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "papersoccer/rules.hpp"

namespace papersoccer::arena {

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::uint64_t kBootstrapSeedSalt = 0x4152454e414349ULL;
constexpr std::size_t kMaxPositionGenerationAttempts = 4096;

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

struct PositionEvaluation {
  Entrant entrant{Entrant::Candidate};
  std::uint64_t seed{};
  Move move{};
  std::uint64_t elapsed_ns{};
  std::optional<SearchStats> stats{};
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

  double score() const noexcept {
    return static_cast<double>(wins) + 0.5 * static_cast<double>(truncations);
  }
};

struct TimingSummary {
  std::size_t decisions{};
  std::uint64_t total_ns{};
  std::uint64_t min_ns{};
  std::uint64_t median_ns{};
  std::uint64_t p95_ns{};
  std::uint64_t max_ns{};
  double median_iterations_per_second{};
  double median_simulated_plies_per_second{};
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
  std::uint64_t rebuild_count_max{};
  std::size_t expansion_saturated_searches{};
};

struct BootstrapInterval {
  std::uint64_t seed{};
  std::size_t samples{};
  double lower_percent{};
  double upper_percent{};
};

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

std::string_view kind_name(BotKind kind) noexcept {
  return kind == BotKind::Random ? "random" : "mcts";
}

void validate_rules(const RulesConfig &rules) {
  if (rules.width < 2 || rules.height < 2) {
    throw std::invalid_argument("arena board width and height must be at least 2");
  }
}

void validate_bot_config(const ArenaBotConfig &config) {
  if (config.kind == BotKind::Mcts) {
    if (config.iterations == 0) {
      throw std::invalid_argument("arena MCTS iterations must be greater than zero");
    }
    if (!std::isfinite(config.exploration) || config.exploration < 0.0) {
      throw std::invalid_argument(
          "arena MCTS exploration must be finite and non-negative");
    }
    if (config.max_nodes < 2) {
      throw std::invalid_argument("arena MCTS max nodes must be at least 2");
    }
  }
}

void validate_common(const RulesConfig &rules, const ArenaBotConfig &candidate,
                     const ArenaBotConfig &reference) {
  validate_rules(rules);
  validate_bot_config(candidate);
  validate_bot_config(reference);
}

std::unique_ptr<Bot> make_arena_bot(const ArenaBotConfig &config,
                                    std::uint64_t seed) {
  if (config.kind == BotKind::Random) {
    return std::make_unique<RandomBot>(seed);
  }

  MctsConfig mcts;
  mcts.seed = seed;
  mcts.iterations = config.iterations;
  mcts.exploration = config.exploration;
  mcts.rollout_policy = config.rollout_policy;
  mcts.reuse_tree = config.reuse_tree;
  mcts.max_nodes = config.max_nodes;
  return std::make_unique<MctsBot>(mcts);
}

std::optional<SearchStats> search_stats(Bot &bot) {
  auto *mcts = dynamic_cast<MctsBot *>(&bot);
  if (mcts == nullptr) {
    return std::nullopt;
  }
  return mcts->last_search_stats();
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
  return DecisionReport{ply, entrant, state.to_move, from, move,
                        elapsed_nanoseconds(start, end), search_stats(bot)};
}

GameReport play_game(std::size_t pair_index, std::size_t game_in_pair,
                     Participant player_one, Participant player_two,
                     const RulesConfig &rules, std::size_t max_plies) {
  GameReport report;
  report.pair_index = pair_index;
  report.game_in_pair = game_in_pair;
  report.player_one = player_one;
  report.player_two = player_two;
  report.decisions.reserve(std::min<std::size_t>(max_plies, 512));

  std::unique_ptr<Bot> player_one_bot =
      make_arena_bot(player_one.config, player_one.seed);
  std::unique_ptr<Bot> player_two_bot =
      make_arena_bot(player_two.config, player_two.seed);
  GameState state = make_initial_state(rules);

  while (!is_terminal(state) && report.decisions.size() < max_plies) {
    const bool one_to_move = state.to_move == Player::One;
    Bot &bot = one_to_move ? *player_one_bot : *player_two_bot;
    const Entrant entrant = one_to_move ? player_one.entrant : player_two.entrant;
    DecisionReport decision = choose_and_measure(
        bot, entrant, report.decisions.size() + 1, state);
    state = apply_move(state, decision.move);
    report.decisions.push_back(std::move(decision));
  }

  report.status = state.status;
  report.plies = report.decisions.size();
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
  return PositionEvaluation{entrant, seed, decision.move, decision.elapsed_ns,
                            decision.stats};
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
  elapsed.reserve(decisions.size());
  throughput.reserve(decisions.size());
  rollout_throughput.reserve(decisions.size());
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
    summary.rebuild_count_max =
        std::max(summary.rebuild_count_max, stats.rebuild_count);
    summary.expansion_saturated_searches += stats.expansion_saturated ? 1U : 0U;
  }
  return summary;
}

Record record_for(const std::vector<GameReport> &games, Entrant entrant,
                  std::optional<Player> color = std::nullopt) {
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
        evaluation.elapsed_ns, evaluation.stats});
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
  if (config.kind == BotKind::Mcts) {
    out << ",\"iterations\":" << config.iterations
        << ",\"exploration\":" << config.exploration << ",\"rollout_policy\":";
    write_string(out, policy_name(config.rollout_policy));
    out << ",\"reuse_tree\":";
    write_bool(out, config.reuse_tree);
    out << ",\"max_nodes\":" << config.max_nodes;
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
  out << ",\"rebuild_count\":" << stats.rebuild_count
      << ",\"expansion_saturated\":";
  write_bool(out, stats.expansion_saturated);
  out << '}';
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
  out << '}';
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
      << summary.median_simulated_plies_per_second << '}';
}

void write_mcts_summary(std::ostream &out, const MctsSummary &summary) {
  out << "{\"searches\":" << summary.searches
      << ",\"iterations\":" << summary.iterations
      << ",\"nodes_sum\":" << summary.nodes_sum
      << ",\"simulated_plies\":" << summary.simulated_plies
      << ",\"total_root_visits_sum\":" << summary.total_root_visits_sum
      << ",\"reused_visits_sum\":" << summary.reused_visits_sum
      << ",\"max_depth\":" << summary.max_depth
      << ",\"proven_nodes_sum\":" << summary.proven_nodes_sum
      << ",\"proven_searches\":" << summary.proven_searches
      << ",\"rebuild_count_max\":" << summary.rebuild_count_max
      << ",\"expansion_saturated_searches\":"
      << summary.expansion_saturated_searches << '}';
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
  out << '}';
}

}  // namespace

std::string run_matches_json(const MatchesConfig &config) {
  validate_common(config.rules, config.candidate, config.reference);
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

  SplitMix64 seeds{config.base_seed};
  std::vector<GameReport> games;
  games.reserve(config.seed_pairs * 2);
  for (std::size_t pair = 0; pair < config.seed_pairs; ++pair) {
    const std::uint64_t candidate_seed = seeds.next();
    const std::uint64_t reference_seed = seeds.next();
    games.push_back(play_game(
        pair, 0,
        Participant{Entrant::Candidate, Player::One, candidate_seed,
                    config.candidate},
        Participant{Entrant::Reference, Player::Two, reference_seed,
                    config.reference},
        config.rules, config.max_plies));
    games.push_back(play_game(
        pair, 1,
        Participant{Entrant::Reference, Player::One, reference_seed,
                    config.reference},
        Participant{Entrant::Candidate, Player::Two, candidate_seed,
                    config.candidate},
        config.rules, config.max_plies));
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
      << ",\"max_plies\":" << config.max_plies
      << ",\"bootstrap_samples\":" << config.bootstrap_samples
      << ",\"candidate\":";
  write_bot_config(out, config.candidate);
  out << ",\"reference\":";
  write_bot_config(out, config.reference);
  out << "},\"games\":[";
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
