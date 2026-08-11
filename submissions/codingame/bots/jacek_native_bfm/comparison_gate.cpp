#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_reference_engine
#include "../rank_4/bot.cpp"
#undef turn_action_v2

#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include "bot.cpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace candidate = papersoccer::jacek_native_bfm;
namespace reference = papersoccer::rank4_reference_engine;

namespace {

constexpr std::uint32_t kFirstOperationalLimitMs = 1'000;
constexpr std::uint32_t kLaterOperationalLimitMs = 200;
constexpr std::uint32_t kFirstHeadroomLimitMs = 900;
constexpr std::uint32_t kLaterHeadroomLimitMs = 180;

struct HarnessConfig {
  int pairs{8};
  std::uint64_t candidate_iterations{256};
  std::size_t candidate_actions{candidate::kMaximumActions};
  std::size_t candidate_partial_paths{candidate::kProductionPartialPaths};
  std::size_t candidate_tree_nodes{candidate::kProductionTreeNodes};
  std::uint64_t reference_nodes{30'000};
  int maximum_turns{200};
  std::vector<int> opening_turns{0, 2, 6, 12};
  std::uint64_t seed{0x64c28ebd2f17a953ULL};
  std::uint32_t candidate_first_ms{};
  std::uint32_t candidate_later_ms{};
  std::uint32_t reference_first_ms{};
  std::uint32_t reference_later_ms{};
  int minimum_candidate_wins{};
  int minimum_wins_per_color{};
  double minimum_wilson_lower{-1.0};
};

struct Opening {
  ps::GameState state;
  std::string transcript;
  std::array<std::uint32_t, 2> player_turns{};
  std::uint64_t seed{};
  int complete_turns{};
};

struct Decision {
  std::string action;
  std::uint64_t work{};
  std::uint64_t child_evaluations{};
  std::uint64_t completed_actions{};
  std::uint64_t duplicate_boundaries{};
  std::uint64_t partial_paths{};
  std::uint64_t tactical_proof_paths{};
  std::uint64_t fifo_extractions{};
  std::uint64_t lifo_extractions{};
  std::uint64_t tactical_classes_found{};
  std::uint64_t tactical_proof_truncations{};
  std::uint64_t truncations{};
  std::uint32_t completed_depth{};
  std::size_t tree_nodes{};
  double milliseconds{};
  double maximum_first_milliseconds{};
  double maximum_later_milliseconds{};
  bool deadline_reached{};
  std::uint64_t headroom_failures{};
  std::uint64_t operational_timeouts{};
};

struct GameResult {
  int winner{-1};
  int turns{};
  Decision candidate_totals;
  Decision reference_totals;
};

struct Summary {
  int candidate_wins{};
  int reference_wins{};
  int unfinished{};
  int candidate_as_player_one{};
  int candidate_as_player_two{};
  int games{};
  int headroom_failure_games{};
  int operational_timeouts{};
  Decision candidate_totals;
  Decision reference_totals;
};

double wilson_lower_bound(int successes, int trials) noexcept {
  if (trials <= 0 || successes < 0 || successes > trials) {
    return 0.0;
  }
  constexpr double kZ95 = 1.959963984540054;
  constexpr double kZ95Squared = kZ95 * kZ95;
  const double count = static_cast<double>(trials);
  const double rate = static_cast<double>(successes) / count;
  const double denominator = 1.0 + kZ95Squared / count;
  const double center = rate + kZ95Squared / (2.0 * count);
  const double radius =
      kZ95 * std::sqrt((rate * (1.0 - rate) +
                        kZ95Squared / (4.0 * count)) /
                       count);
  return std::max(0.0, (center - radius) / denominator);
}

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) noexcept {
  return player == ps::Player::One ? 0 : 1;
}

std::uint64_t splitmix64(std::uint64_t &state) noexcept {
  state += 0x9e3779b97f4a7c15ULL;
  std::uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::size_t random_index(std::uint64_t &state, std::size_t bound) noexcept {
  if (bound < 2) return 0;
  const std::uint64_t unsigned_bound = static_cast<std::uint64_t>(bound);
  const std::uint64_t threshold =
      static_cast<std::uint64_t>(-unsigned_bound) % unsigned_bound;
  for (;;) {
    const std::uint64_t value = splitmix64(state);
    if (value >= threshold) {
      return static_cast<std::size_t>(value % unsigned_bound);
    }
  }
}

void append_transcript(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) {
    transcript.push_back('/');
  }
  transcript.append(action);
}

std::string procedural_turn(ps::GameState &state, std::uint64_t &seed) {
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("cannot construct a turn from terminal state");
  }
  const ps::Player mover = state.to_move;
  std::string encoded;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    if (legal.empty()) {
      throw std::logic_error("in-progress opening state has no legal move");
    }
    const ps::Move move = legal[random_index(seed, legal.size())];
    encoded.push_back(candidate::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
    if (encoded.size() > 316U) {
      throw std::logic_error("procedural opening turn exceeded board edges");
    }
  }
  return encoded;
}

Opening make_opening(std::uint64_t base_seed, int complete_turns) {
  for (std::uint64_t retry = 0; retry < 4'096; ++retry) {
    Opening opening{ps::make_initial_state(codingame_rules()), {}, {},
                    base_seed + retry * 0xd1b54a32d192ed03ULL,
                    complete_turns};
    std::uint64_t random = opening.seed;
    bool valid = true;
    for (int turn = 0; turn < complete_turns; ++turn) {
      if (ps::is_terminal(opening.state)) {
        valid = false;
        break;
      }
      const int mover = player_id(opening.state.to_move);
      const std::string action = procedural_turn(opening.state, random);
      append_transcript(opening.transcript, action);
      ++opening.player_turns[static_cast<std::size_t>(mover)];
    }
    if (valid && !ps::is_terminal(opening.state)) {
      return opening;
    }
  }
  throw std::runtime_error("could not generate a nonterminal procedural opening");
}

std::uint32_t clock_budget(std::uint32_t turns, std::uint32_t first,
                           std::uint32_t later) noexcept {
  return turns == 0 ? first : later;
}

bool is_operational_timeout(double elapsed, std::uint32_t turns,
                            std::uint32_t configured_budget) noexcept {
  if (configured_budget == 0) {
    return false;
  }
  const std::uint32_t limit =
      turns == 0 ? kFirstOperationalLimitMs : kLaterOperationalLimitMs;
  return elapsed >= static_cast<double>(limit);
}

bool is_headroom_failure(double elapsed, std::uint32_t turns,
                         std::uint32_t configured_budget) noexcept {
  if (configured_budget == 0) {
    return false;
  }
  const std::uint32_t limit =
      turns == 0 ? kFirstHeadroomLimitMs : kLaterHeadroomLimitMs;
  return elapsed >= static_cast<double>(limit);
}

Decision choose_candidate(ps::GameState &state,
                          const HarnessConfig &config,
                          std::uint32_t prior_turns,
                          std::uint64_t decision_seed) {
  candidate::SearchConfig search_config;
  search_config.max_actions = config.candidate_actions;
  search_config.max_partial_paths = config.candidate_partial_paths;
  search_config.max_tree_nodes = config.candidate_tree_nodes;
  search_config.max_expansions = config.candidate_iterations;
  search_config.shuffle_seed = decision_seed;
  const std::uint32_t time_ms =
      clock_budget(prior_turns, config.candidate_first_ms,
                   config.candidate_later_ms);
  if (time_ms != 0) {
    search_config.absolute_deadline = candidate::SearchClock::now() +
                                      std::chrono::milliseconds(time_ms);
  }
  const auto started = std::chrono::steady_clock::now();
  const candidate::SearchResult result =
      candidate::choose_complete_turn(static_cast<const ps::GameState &>(state),
                                      search_config);
  const auto finished = std::chrono::steady_clock::now();
  ps::GameState next = state;
  candidate::apply_encoded_turn(next, result.encoded);
  state = std::move(next);

  Decision decision;
  decision.action = result.encoded;
  decision.work = result.stats.expansions;
  decision.child_evaluations = result.stats.child_evaluations;
  decision.completed_actions = result.stats.completed_actions;
  decision.duplicate_boundaries = result.stats.duplicate_boundaries;
  decision.partial_paths = result.stats.generator_partial_paths;
  decision.tactical_proof_paths = result.stats.proof_paths;
  decision.fifo_extractions = result.stats.generator_fifo_extractions;
  decision.lifo_extractions = result.stats.generator_lifo_extractions;
  decision.tactical_classes_found = result.stats.proof_classes;
  decision.tactical_proof_truncations =
      result.stats.proof_truncations;
  decision.truncations = result.stats.generator_truncations;
  decision.tree_nodes = result.stats.tree_nodes;
  decision.deadline_reached = result.stats.deadline_reached;
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(finished - started).count();
  if (prior_turns == 0) {
    decision.maximum_first_milliseconds = decision.milliseconds;
  } else {
    decision.maximum_later_milliseconds = decision.milliseconds;
  }
  decision.headroom_failures =
      is_headroom_failure(decision.milliseconds, prior_turns, time_ms) ? 1U
                                                                       : 0U;
  decision.operational_timeouts =
      is_operational_timeout(decision.milliseconds, prior_turns, time_ms) ? 1U
                                                                          : 0U;
  return decision;
}

Decision choose_reference(ps::GameState &state, std::string_view transcript,
                          const HarnessConfig &config,
                          std::uint32_t prior_turns) {
  const std::uint32_t time_ms =
      clock_budget(prior_turns, config.reference_first_ms,
                   config.reference_later_ms);
  const auto started = std::chrono::steady_clock::now();
  std::string encoded;
  reference::SearchStats stats;
  if (!reference::try_replay_correction(
          state, player_id(state.to_move), transcript, encoded)) {
    reference::SearchConfig search_config;
    search_config.max_nodes = config.reference_nodes;
    search_config.replay_value_blend_percent = 15;
    search_config.teacher_residual_weight_percent = 100;
    if (time_ms != 0) {
      search_config.absolute_deadline = reference::SearchClock::now() +
                                        std::chrono::milliseconds(time_ms);
    }
    reference::CompleteTurnSearch search(state, search_config);
    const std::vector<ps::Move> moves = search.run();
    stats = search.stats();
    ps::GameState next = state;
    const ps::Player mover = next.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(next) || next.to_move != mover) {
        throw std::logic_error("reference action extends after turn end");
      }
      encoded.push_back(candidate::encode_direction(next.ball, move.to));
      next = ps::apply_move(next, move);
    }
    state = std::move(next);
  }
  const auto finished = std::chrono::steady_clock::now();
  if (encoded.empty()) {
    throw std::logic_error("reference returned an empty action");
  }
  Decision decision;
  decision.action = std::move(encoded);
  decision.work = stats.nodes;
  decision.completed_actions = stats.completed_actions;
  decision.completed_depth = stats.completed_turn_depth;
  decision.deadline_reached = stats.budget_exhausted;
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(finished - started).count();
  if (prior_turns == 0) {
    decision.maximum_first_milliseconds = decision.milliseconds;
  } else {
    decision.maximum_later_milliseconds = decision.milliseconds;
  }
  decision.headroom_failures =
      is_headroom_failure(decision.milliseconds, prior_turns, time_ms) ? 1U
                                                                       : 0U;
  decision.operational_timeouts =
      is_operational_timeout(decision.milliseconds, prior_turns, time_ms) ? 1U
                                                                          : 0U;
  return decision;
}

void add_decision(Decision &total, const Decision &decision) {
  total.work += decision.work;
  total.child_evaluations += decision.child_evaluations;
  total.completed_actions += decision.completed_actions;
  total.duplicate_boundaries += decision.duplicate_boundaries;
  total.partial_paths += decision.partial_paths;
  total.tactical_proof_paths += decision.tactical_proof_paths;
  total.fifo_extractions += decision.fifo_extractions;
  total.lifo_extractions += decision.lifo_extractions;
  total.tactical_classes_found += decision.tactical_classes_found;
  total.tactical_proof_truncations += decision.tactical_proof_truncations;
  total.truncations += decision.truncations;
  total.completed_depth += decision.completed_depth;
  total.tree_nodes = std::max(total.tree_nodes, decision.tree_nodes);
  total.milliseconds += decision.milliseconds;
  total.maximum_first_milliseconds = std::max(
      total.maximum_first_milliseconds, decision.maximum_first_milliseconds);
  total.maximum_later_milliseconds = std::max(
      total.maximum_later_milliseconds, decision.maximum_later_milliseconds);
  total.deadline_reached = total.deadline_reached || decision.deadline_reached;
  total.headroom_failures += decision.headroom_failures;
  total.operational_timeouts += decision.operational_timeouts;
}

GameResult play(const Opening &opening, int candidate_player,
                const HarnessConfig &config) {
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  std::array<std::uint32_t, 2> player_turns = opening.player_turns;
  GameResult result;
  for (; result.turns < config.maximum_turns && !ps::is_terminal(state);
       ++result.turns) {
    const int mover = player_id(state.to_move);
    Decision decision;
    if (mover == candidate_player) {
      const std::uint64_t decision_seed =
          config.seed ^ opening.seed ^
          (static_cast<std::uint64_t>(result.turns + 1) *
           0x9e3779b97f4a7c15ULL);
      decision = choose_candidate(
          state, config, player_turns[static_cast<std::size_t>(mover)],
          decision_seed);
      add_decision(result.candidate_totals, decision);
    } else {
      decision = choose_reference(
          state, transcript, config,
          player_turns[static_cast<std::size_t>(mover)]);
      add_decision(result.reference_totals, decision);
    }
    append_transcript(transcript, decision.action);
    ++player_turns[static_cast<std::size_t>(mover)];
  }
  if (const auto winner = ps::winner(state); winner.has_value()) {
    result.winner = player_id(*winner);
  }
  return result;
}

void accumulate(Summary &summary, const GameResult &game,
                int candidate_player) {
  ++summary.games;
  add_decision(summary.candidate_totals, game.candidate_totals);
  add_decision(summary.reference_totals, game.reference_totals);
  if (game.candidate_totals.headroom_failures != 0 ||
      game.reference_totals.headroom_failures != 0) {
    ++summary.headroom_failure_games;
  }
  if (game.candidate_totals.operational_timeouts != 0 ||
      game.reference_totals.operational_timeouts != 0) {
    ++summary.operational_timeouts;
  }
  if (game.winner < 0) {
    ++summary.unfinished;
  } else if (game.winner == candidate_player) {
    ++summary.candidate_wins;
    if (candidate_player == 0) {
      ++summary.candidate_as_player_one;
    } else {
      ++summary.candidate_as_player_two;
    }
  } else {
    ++summary.reference_wins;
  }
}

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view label) {
  Integer value{};
  const char *first = text.data();
  const char *last = first + text.size();
  const auto [end, error] = std::from_chars(first, last, value);
  if (error != std::errc{} || end != last) {
    throw std::invalid_argument(std::string(label) + " must be an integer");
  }
  return value;
}

double parse_probability(std::string_view text, std::string_view label) {
  double value{};
  const char *first = text.data();
  const char *last = first + text.size();
  const auto [end, error] =
      std::from_chars(first, last, value, std::chars_format::general);
  if (error != std::errc{} || end != last || !std::isfinite(value) ||
      value < 0.0 || value > 1.0) {
    throw std::invalid_argument(std::string(label) +
                                " must be a probability in [0,1]");
  }
  return value;
}

std::vector<int> parse_opening_turns(std::string_view text) {
  std::vector<int> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end =
        comma == std::string_view::npos ? text.size() : comma;
    if (end == start) {
      throw std::invalid_argument("opening turns contain an empty value");
    }
    const int turns = parse_integer<int>(text.substr(start, end - start),
                                         "opening turns");
    if (turns < 0) {
      throw std::invalid_argument("opening turns must be nonnegative");
    }
    result.push_back(turns);
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

void print_help() {
  std::cout
      << "Usage: comparison_gate [options]\n"
      << "  --pairs N                    openings per requested depth\n"
      << "                               (two color-swapped games each)\n"
      << "  --candidate-iterations N     BFM expansion cap\n"
      << "  --candidate-actions N        complete actions per expansion\n"
      << "  --candidate-partial-paths N  generator partial-path cap\n"
      << "  --candidate-tree-nodes N     BFM tree-node cap\n"
      << "  --reference-nodes N          external rank-4 node cap\n"
      << "  --maximum-turns N            decisions after each opening\n"
      << "  --opening-turns A,B,...      procedural opening depths\n"
      << "  --seed N                     deterministic batch seed\n"
      << "  --equal-clock                both bots at 800/155 ms\n"
      << "  --candidate-first-ms N --candidate-later-ms N\n"
      << "  --reference-first-ms N --reference-later-ms N\n"
      << "  --minimum-candidate-wins N  optional total wins floor\n"
      << "  --minimum-wins-per-color N  optional wins floor in each color\n"
      << "  --minimum-wilson-lower P    optional 95% Wilson lower bound\n";
}

HarnessConfig parse_arguments(int argc, char **argv) {
  HarnessConfig config;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help") {
      print_help();
      std::exit(0);
    }
    if (argument == "--equal-clock") {
      config.candidate_first_ms = candidate::kFirstSearchTimeMs;
      config.candidate_later_ms = candidate::kLaterSearchTimeMs;
      config.reference_first_ms = candidate::kFirstSearchTimeMs;
      config.reference_later_ms = candidate::kLaterSearchTimeMs;
      config.candidate_iterations = candidate::kMaximumExpansions;
      config.candidate_actions = candidate::kMaximumActions;
      config.candidate_partial_paths = candidate::kProductionPartialPaths;
      config.candidate_tree_nodes = candidate::kProductionTreeNodes;
      config.reference_nodes = reference::kMaximumNodes;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(argument) + " needs a value");
    }
    const std::string_view value = argv[++index];
    if (argument == "--pairs") {
      config.pairs = parse_integer<int>(value, "pairs");
    } else if (argument == "--candidate-iterations") {
      config.candidate_iterations =
          parse_integer<std::uint64_t>(value, "candidate iterations");
    } else if (argument == "--candidate-actions") {
      config.candidate_actions =
          parse_integer<std::size_t>(value, "candidate actions");
    } else if (argument == "--candidate-partial-paths") {
      config.candidate_partial_paths =
          parse_integer<std::size_t>(value, "candidate partial paths");
    } else if (argument == "--candidate-tree-nodes") {
      config.candidate_tree_nodes =
          parse_integer<std::size_t>(value, "candidate tree nodes");
    } else if (argument == "--reference-nodes") {
      config.reference_nodes =
          parse_integer<std::uint64_t>(value, "reference nodes");
    } else if (argument == "--maximum-turns") {
      config.maximum_turns = parse_integer<int>(value, "maximum turns");
    } else if (argument == "--opening-turns") {
      config.opening_turns = parse_opening_turns(value);
    } else if (argument == "--seed") {
      config.seed = parse_integer<std::uint64_t>(value, "seed");
    } else if (argument == "--candidate-first-ms") {
      config.candidate_first_ms =
          parse_integer<std::uint32_t>(value, "candidate first ms");
    } else if (argument == "--candidate-later-ms") {
      config.candidate_later_ms =
          parse_integer<std::uint32_t>(value, "candidate later ms");
    } else if (argument == "--reference-first-ms") {
      config.reference_first_ms =
          parse_integer<std::uint32_t>(value, "reference first ms");
    } else if (argument == "--reference-later-ms") {
      config.reference_later_ms =
          parse_integer<std::uint32_t>(value, "reference later ms");
    } else if (argument == "--minimum-candidate-wins") {
      config.minimum_candidate_wins =
          parse_integer<int>(value, "minimum candidate wins");
    } else if (argument == "--minimum-wins-per-color") {
      config.minimum_wins_per_color =
          parse_integer<int>(value, "minimum wins per color");
    } else if (argument == "--minimum-wilson-lower") {
      config.minimum_wilson_lower =
          parse_probability(value, "minimum Wilson lower bound");
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (config.pairs <= 0 || config.maximum_turns <= 0 ||
      config.opening_turns.empty() || config.candidate_iterations == 0 ||
      config.candidate_iterations > candidate::kMaximumExpansions ||
      config.candidate_actions == 0 ||
      config.candidate_actions > candidate::kMaximumActions ||
      config.candidate_partial_paths == 0 ||
      config.candidate_partial_paths > candidate::kMaximumPartialPaths ||
      config.candidate_tree_nodes < 2 ||
      config.candidate_tree_nodes > candidate::kMaximumTreeNodes ||
      config.reference_nodes == 0 || config.minimum_candidate_wins < 0 ||
      config.minimum_wins_per_color < 0) {
    throw std::invalid_argument("comparison limits are outside supported ranges");
  }
  const std::size_t paired_openings =
      static_cast<std::size_t>(config.pairs) * config.opening_turns.size();
  if (static_cast<std::size_t>(config.minimum_wins_per_color) >
      paired_openings) {
    throw std::invalid_argument(
        "minimum wins per color exceeds games available in that color");
  }
  if (static_cast<std::size_t>(config.minimum_candidate_wins) >
      2U * paired_openings) {
    throw std::invalid_argument(
        "minimum candidate wins exceeds games in the configured batch");
  }
  return config;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const HarnessConfig config = parse_arguments(argc, argv);
    Summary summary;
    std::uint64_t opening_index = 0;
    for (const int turns : config.opening_turns) {
      for (int pair = 0; pair < config.pairs; ++pair) {
        const std::uint64_t seed =
            config.seed + (++opening_index * 0x9e3779b97f4a7c15ULL);
        const Opening opening = make_opening(seed, turns);
        std::cout << "opening turns=" << turns << " pair=" << pair
                  << " seed=" << opening.seed;
        for (int candidate_player = 0; candidate_player < 2;
             ++candidate_player) {
          const GameResult game = play(opening, candidate_player, config);
          accumulate(summary, game, candidate_player);
          std::cout << " c" << candidate_player << '=' << game.winner
                    << ':' << game.turns
                    << ":ce" << game.candidate_totals.work
                    << ":rn" << game.reference_totals.work;
        }
        std::cout << '\n';
      }
    }
    const int finished_games =
        summary.candidate_wins + summary.reference_wins;
    const double candidate_win_rate =
        finished_games == 0
            ? 0.0
            : static_cast<double>(summary.candidate_wins) / finished_games;
    const double candidate_wilson_lower =
        wilson_lower_bound(summary.candidate_wins, finished_games);
    const bool color_gate_passed =
        summary.candidate_as_player_one >= config.minimum_wins_per_color &&
        summary.candidate_as_player_two >= config.minimum_wins_per_color;
    const bool total_wins_gate_passed =
        summary.candidate_wins >= config.minimum_candidate_wins;
    const bool wilson_gate_passed =
        config.minimum_wilson_lower < 0.0 ||
        candidate_wilson_lower >= config.minimum_wilson_lower;
    std::cout << "summary candidate=" << summary.candidate_wins
              << " reference=" << summary.reference_wins
              << " unfinished=" << summary.unfinished
              << " candidate_player_one=" << summary.candidate_as_player_one
              << " candidate_player_two=" << summary.candidate_as_player_two
              << " paired_openings=" << opening_index
              << " games=" << summary.games
              << " candidate_win_rate=" << candidate_win_rate
              << " candidate_wilson95_lower=" << candidate_wilson_lower
              << " minimum_candidate_wins="
              << config.minimum_candidate_wins
              << " minimum_wins_per_color="
              << config.minimum_wins_per_color
              << " minimum_wilson_lower=" << config.minimum_wilson_lower
              << " color_gate_passed=" << color_gate_passed
              << " total_wins_gate_passed=" << total_wins_gate_passed
              << " wilson_gate_passed=" << wilson_gate_passed
              << " candidate_expansions=" << summary.candidate_totals.work
              << " candidate_child_evaluations="
              << summary.candidate_totals.child_evaluations
              << " candidate_completed_actions="
              << summary.candidate_totals.completed_actions
              << " candidate_boundary_duplicates="
              << summary.candidate_totals.duplicate_boundaries
              << " candidate_partial_paths="
              << summary.candidate_totals.partial_paths
              << " candidate_tactical_proof_paths="
              << summary.candidate_totals.tactical_proof_paths
              << " candidate_fifo="
              << summary.candidate_totals.fifo_extractions
              << " candidate_lifo="
              << summary.candidate_totals.lifo_extractions
              << " candidate_tactical_classes_found="
              << summary.candidate_totals.tactical_classes_found
              << " candidate_tactical_proof_truncations="
              << summary.candidate_totals.tactical_proof_truncations
              << " candidate_truncations="
              << summary.candidate_totals.truncations
              << " candidate_max_tree="
              << summary.candidate_totals.tree_nodes
              << " candidate_ms=" << summary.candidate_totals.milliseconds
              << " candidate_max_first_ms="
              << summary.candidate_totals.maximum_first_milliseconds
              << " candidate_max_later_ms="
              << summary.candidate_totals.maximum_later_milliseconds
              << " reference_nodes=" << summary.reference_totals.work
              << " reference_depth_sum="
              << summary.reference_totals.completed_depth
              << " reference_ms=" << summary.reference_totals.milliseconds
              << " reference_max_first_ms="
              << summary.reference_totals.maximum_first_milliseconds
              << " reference_max_later_ms="
              << summary.reference_totals.maximum_later_milliseconds
              << " candidate_headroom_failures="
              << summary.candidate_totals.headroom_failures
              << " reference_headroom_failures="
              << summary.reference_totals.headroom_failures
              << " headroom_failure_games=" << summary.headroom_failure_games
              << " candidate_operational_timeouts="
              << summary.candidate_totals.operational_timeouts
              << " reference_operational_timeouts="
              << summary.reference_totals.operational_timeouts
              << " operational_timeouts=" << summary.operational_timeouts
              << " candidate_limits=" << config.candidate_iterations << ':'
              << config.candidate_actions << ':'
              << config.candidate_partial_paths << ':'
              << config.candidate_tree_nodes
              << " candidate_profile=" << config.candidate_first_ms << '/'
              << config.candidate_later_ms
              << " reference_profile=" << config.reference_first_ms << '/'
              << config.reference_later_ms << '\n';
    return summary.unfinished == 0 && summary.headroom_failure_games == 0 &&
                   summary.operational_timeouts == 0 && color_gate_passed &&
                   total_wins_gate_passed && wilson_gate_passed
               ? 0
               : 1;
  } catch (const std::exception &error) {
    std::cerr << "jacek_native_bfm comparison failure: " << error.what()
              << '\n';
    return 1;
  }
}
