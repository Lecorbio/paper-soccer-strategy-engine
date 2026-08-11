#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_reference_engine
#include "../rank_4/bot.cpp"
#undef turn_action_v2

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 candidate_engine
#include "bot.cpp"
#undef turn_action_v2

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace reference = papersoccer::rank4_reference_engine;
namespace candidate = papersoccer::candidate_engine;

namespace {

constexpr std::uint64_t kOpeningSeedSalt = 0x8f3f73b5cf1c9d6bULL;
constexpr std::uint64_t kOpeningRetryStride = 0x9e3779b97f4a7c15ULL;
constexpr std::uint32_t kFirstOperationalLimitMs = 1'000;
constexpr std::uint32_t kLaterOperationalLimitMs = 200;

struct HarnessConfig {
  int pairs_per_depth{8};
  std::uint64_t candidate_work{30'000};
  std::uint64_t reference_nodes{30'000};
  int maximum_turns{200};
  std::uint64_t batch_start{};
  std::uint64_t batch_count{1};
  std::vector<int> opening_turns{2, 6, 12, 20};
  std::uint32_t candidate_first_time_ms{};
  std::uint32_t candidate_later_time_ms{};
  std::uint32_t reference_first_time_ms{};
  std::uint32_t reference_later_time_ms{};
  bool candidate_replay_corrections{true};
  bool reference_replay_corrections{true};
  std::size_t candidate_max_actions{candidate::kMaximumActions};
  std::size_t candidate_nonroot_actions{};
  double candidate_exploration{candidate::kExplorationConstant};
  double candidate_fpu{candidate::kFirstPlayUrgency};
  double candidate_final_visit_weight{};
  int candidate_replay_blend{15};
  int candidate_residual_weight{100};
  bool candidate_root_only{};
  std::uint32_t position_time_ms{};
  std::optional<std::string> position{};
};

struct Decision {
  std::string action;
  std::uint64_t work{};
  std::uint64_t tree_nodes{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t completed_actions{};
  std::uint64_t generator_partial_paths{};
  std::uint64_t generator_duplicates{};
  std::uint64_t tactical_actions{};
  std::uint64_t generator_truncations{};
  std::uint32_t max_deque_size{};
  std::uint32_t max_action_edges{};
  std::uint64_t reference_nodes{};
  std::uint32_t reference_completed_depth{};
  std::uint32_t reference_attempted_depth{};
  int root_score{};
  double milliseconds{};
  bool budget_exhausted{};
  bool deadline_reached{};
  bool node_cap_reached{};
  bool operational_timeout{};
  bool first_response{};
  bool searched{};
};

struct CandidateTotals {
  std::uint64_t work{};
  std::uint64_t tree_nodes{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t completed_actions{};
  std::uint64_t generator_partial_paths{};
  std::uint64_t generator_duplicates{};
  std::uint64_t tactical_actions{};
  std::uint64_t generator_truncations{};
  std::uint64_t searches{};
  std::uint64_t exhaustions{};
  std::uint64_t deadlines{};
  std::uint64_t node_caps{};
  std::uint64_t operational_timeouts{};
  double milliseconds{};
  std::uint64_t maximum_tree_nodes{};
  std::uint32_t maximum_deque_size{};
  std::uint32_t maximum_action_edges{};
  double maximum_first_milliseconds{};
  double maximum_later_milliseconds{};
};

struct ReferenceTotals {
  std::uint64_t nodes{};
  std::uint64_t completed_depth{};
  std::uint64_t attempted_depth{};
  std::uint64_t searches{};
  std::uint64_t exhaustions{};
  std::uint64_t deadlines{};
  std::uint64_t node_caps{};
  std::uint64_t operational_timeouts{};
  double milliseconds{};
  double maximum_first_milliseconds{};
  double maximum_later_milliseconds{};
};

struct Opening {
  ps::GameState state;
  std::string transcript;
  std::string label;
  std::string batch_tag;
  std::uint64_t seed{};
  bool seeded{};
};

struct GameResult {
  int winner{-1};
  int turns{};
  int candidate_last_root_score{};
  int reference_last_root_score{};
  CandidateTotals candidate{};
  ReferenceTotals reference{};
};

struct Summary {
  int games{};
  int candidate_wins{};
  int reference_wins{};
  int unfinished{};
  std::array<int, 2> color_games{};
  std::array<int, 2> candidate_color_wins{};
  std::array<int, 2> reference_color_wins{};
  std::array<int, 2> color_unfinished{};
  CandidateTotals candidate{};
  ReferenceTotals reference{};
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}
int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) {
    transcript.push_back('/');
  }
  transcript.append(action);
}

template <typename UInt>
UInt parse_unsigned(std::string_view raw, std::string_view label,
                    bool allow_zero = false) {
  static_assert(std::is_unsigned_v<UInt>);
  UInt value{};
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (raw.empty() || raw.front() == '-' || error != std::errc{} ||
      position != end || (!allow_zero && value == 0)) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return value;
}

int parse_positive_int(std::string_view raw, std::string_view label) {
  const unsigned int value = parse_unsigned<unsigned int>(raw, label);
  if (value > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(label) + " is too large");
  }
  return static_cast<int>(value);
}

int parse_nonnegative_int(std::string_view raw, std::string_view label) {
  const unsigned int value =
      parse_unsigned<unsigned int>(raw, label, true);
  if (value > static_cast<unsigned int>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument(std::string(label) + " is too large");
  }
  return static_cast<int>(value);
}

double parse_finite_double(std::string_view raw, std::string_view label) {
  const std::string owned(raw);
  std::size_t consumed = 0;
  double value{};
  try {
    value = std::stod(owned, &consumed);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  if (consumed != owned.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return value;
}

std::vector<int> parse_opening_turns(std::string_view raw) {
  std::vector<int> turns;
  std::size_t begin = 0;
  while (begin <= raw.size()) {
    const std::size_t separator = raw.find(',', begin);
    const std::size_t end =
        separator == std::string_view::npos ? raw.size() : separator;
    const int value = parse_nonnegative_int(
        raw.substr(begin, end - begin), "opening turn count");
    if (std::find(turns.begin(), turns.end(), value) != turns.end()) {
      throw std::invalid_argument("opening turn counts must be unique");
    }
    turns.push_back(value);
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (turns.empty() || turns.size() > 64U) {
    throw std::invalid_argument("opening turn list is invalid");
  }
  return turns;
}

bool parse_bool(std::string_view raw, std::string_view label) {
  const unsigned int value =
      parse_unsigned<unsigned int>(raw, label, true);
  if (value > 1U) {
    throw std::invalid_argument(std::string(label) + " must be 0 or 1");
  }
  return value != 0U;
}

void apply_environment(HarnessConfig &config) {
  const auto unsigned_environment = [](const char *name,
                                       std::uint64_t &destination,
                                       bool allow_zero = false) {
    if (const char *raw = std::getenv(name)) {
      destination = parse_unsigned<std::uint64_t>(raw, name, allow_zero);
    }
  };
  const auto time_environment = [](const char *name,
                                   std::uint32_t &destination) {
    if (const char *raw = std::getenv(name)) {
      destination = parse_unsigned<std::uint32_t>(raw, name, true);
    }
  };
  const auto int_environment = [](const char *name, int &destination) {
    if (const char *raw = std::getenv(name)) {
      destination = parse_positive_int(raw, name);
    }
  };
  const auto bool_environment = [](const char *name, bool &destination) {
    if (const char *raw = std::getenv(name)) {
      const unsigned int value =
          parse_unsigned<unsigned int>(raw, name, true);
      if (value > 1U) {
        throw std::invalid_argument(std::string(name) + " must be 0 or 1");
      }
      destination = value != 0U;
    }
  };

  int_environment("RANK_4_FULLTURN_BFM_PAIRS_PER_DEPTH",
                  config.pairs_per_depth);
  unsigned_environment("RANK_4_FULLTURN_BFM_CANDIDATE_WORK",
                       config.candidate_work);
  unsigned_environment("RANK_4_FULLTURN_BFM_REFERENCE_NODES",
                       config.reference_nodes);
  int_environment("RANK_4_FULLTURN_BFM_MAX_TURNS", config.maximum_turns);
  unsigned_environment("RANK_4_FULLTURN_BFM_BATCH_START",
                       config.batch_start, true);
  unsigned_environment("RANK_4_FULLTURN_BFM_BATCH_COUNT",
                       config.batch_count);
  time_environment("RANK_4_FULLTURN_BFM_CANDIDATE_FIRST_MS",
                   config.candidate_first_time_ms);
  time_environment("RANK_4_FULLTURN_BFM_CANDIDATE_LATER_MS",
                   config.candidate_later_time_ms);
  time_environment("RANK_4_FULLTURN_BFM_REFERENCE_FIRST_MS",
                   config.reference_first_time_ms);
  time_environment("RANK_4_FULLTURN_BFM_REFERENCE_LATER_MS",
                   config.reference_later_time_ms);
  bool_environment("RANK_4_FULLTURN_BFM_CANDIDATE_REPLAY",
                   config.candidate_replay_corrections);
  bool_environment("RANK_4_FULLTURN_BFM_REFERENCE_REPLAY",
                   config.reference_replay_corrections);
}

void print_usage() {
  std::cout
      << "usage: comparison_gate [pairs shared-work max-turns]\n"
         "       comparison_gate [pairs candidate-work reference-nodes "
         "max-turns [batch-start [batch-count]]]\n"
         "       comparison_gate [options]\n"
         "options:\n"
         "  --pairs-per-depth N\n"
         "  --candidate-work N\n"
         "  --reference-nodes N\n"
         "  --max-turns N\n"
         "  --batch-start N\n"
         "  --batch-count N\n"
         "  --opening-turns N[,N...]\n"
         "  --candidate-first-ms N\n"
         "  --candidate-later-ms N\n"
         "  --reference-first-ms N\n"
         "  --reference-later-ms N\n"
         "  --candidate-replay 0|1\n"
         "  --reference-replay 0|1\n"
         "  --candidate-max-actions N\n"
         "  --candidate-nonroot-actions N (0 reuses root cap)\n"
         "  --candidate-exploration X\n"
         "  --candidate-fpu X\n"
         "  --candidate-final-visit-weight X\n"
         "  --candidate-replay-blend N\n"
         "  --candidate-residual-weight N\n"
         "  --candidate-root-only 0|1\n"
         "  --position-time-ms N\n"
         "  --position TRANSCRIPT\n";
}

HarnessConfig parse_options(int argc, char **argv) {
  HarnessConfig config;
  apply_environment(config);

  std::vector<std::string_view> positional;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help" || argument == "-h") {
      print_usage();
      std::exit(0);
    }
    if (!argument.starts_with("--")) {
      positional.push_back(argument);
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " +
                                  std::string(argument));
    }
    const std::string_view value(argv[++index]);
    if (argument == "--pairs-per-depth") {
      config.pairs_per_depth = parse_positive_int(value, argument);
    } else if (argument == "--candidate-work") {
      config.candidate_work =
          parse_unsigned<std::uint64_t>(value, argument);
    } else if (argument == "--reference-nodes") {
      config.reference_nodes =
          parse_unsigned<std::uint64_t>(value, argument);
    } else if (argument == "--max-turns") {
      config.maximum_turns = parse_positive_int(value, argument);
    } else if (argument == "--batch-start") {
      config.batch_start =
          parse_unsigned<std::uint64_t>(value, argument, true);
    } else if (argument == "--batch-count") {
      config.batch_count =
          parse_unsigned<std::uint64_t>(value, argument);
    } else if (argument == "--opening-turns") {
      config.opening_turns = parse_opening_turns(value);
    } else if (argument == "--candidate-first-ms") {
      config.candidate_first_time_ms =
          parse_unsigned<std::uint32_t>(value, argument, true);
    } else if (argument == "--candidate-later-ms") {
      config.candidate_later_time_ms =
          parse_unsigned<std::uint32_t>(value, argument, true);
    } else if (argument == "--reference-first-ms") {
      config.reference_first_time_ms =
          parse_unsigned<std::uint32_t>(value, argument, true);
    } else if (argument == "--reference-later-ms") {
      config.reference_later_time_ms =
          parse_unsigned<std::uint32_t>(value, argument, true);
    } else if (argument == "--candidate-replay") {
      config.candidate_replay_corrections = parse_bool(value, argument);
    } else if (argument == "--reference-replay") {
      config.reference_replay_corrections = parse_bool(value, argument);
    } else if (argument == "--candidate-max-actions") {
      config.candidate_max_actions =
          parse_unsigned<std::size_t>(value, argument);
      if (config.candidate_max_actions > candidate::kMaximumActions) {
        throw std::invalid_argument("candidate action cap exceeds 250");
      }
    } else if (argument == "--candidate-nonroot-actions") {
      const unsigned int cap =
          parse_unsigned<unsigned int>(value, argument, true);
      if (cap > candidate::kMaximumActions) {
        throw std::invalid_argument("candidate non-root cap exceeds 250");
      }
      config.candidate_nonroot_actions = cap;
    } else if (argument == "--candidate-exploration") {
      config.candidate_exploration = parse_finite_double(value, argument);
      if (config.candidate_exploration < 0.0) {
        throw std::invalid_argument("candidate exploration must be nonnegative");
      }
    } else if (argument == "--candidate-fpu") {
      config.candidate_fpu = parse_finite_double(value, argument);
    } else if (argument == "--candidate-final-visit-weight") {
      config.candidate_final_visit_weight =
          parse_finite_double(value, argument);
      if (config.candidate_final_visit_weight < 0.0) {
        throw std::invalid_argument("final visit weight must be nonnegative");
      }
    } else if (argument == "--candidate-replay-blend") {
      config.candidate_replay_blend = parse_nonnegative_int(value, argument);
      if (config.candidate_replay_blend > 100) {
        throw std::invalid_argument("candidate replay blend exceeds 100");
      }
    } else if (argument == "--candidate-residual-weight") {
      config.candidate_residual_weight =
          parse_nonnegative_int(value, argument);
      if (config.candidate_residual_weight > 100) {
        throw std::invalid_argument("candidate residual weight exceeds 100");
      }
    } else if (argument == "--candidate-root-only") {
      config.candidate_root_only = parse_bool(value, argument);
    } else if (argument == "--position-time-ms") {
      config.position_time_ms =
          parse_unsigned<std::uint32_t>(value, argument, true);
    } else if (argument == "--position") {
      config.position = std::string(value);
    } else {
      throw std::invalid_argument("unknown option: " +
                                  std::string(argument));
    }
  }

  if (positional.size() > 6U) {
    throw std::invalid_argument("too many positional arguments");
  }
  if (!positional.empty()) {
    config.pairs_per_depth =
        parse_positive_int(positional[0], "pairs per depth");
  }
  if (positional.size() == 2U) {
    const std::uint64_t shared =
        parse_unsigned<std::uint64_t>(positional[1], "shared work");
    config.candidate_work = shared;
    config.reference_nodes = shared;
  } else if (positional.size() == 3U) {
    // Preserve the established `pairs shared-work max-turns` smoke command.
    const std::uint64_t shared =
        parse_unsigned<std::uint64_t>(positional[1], "shared work");
    config.candidate_work = shared;
    config.reference_nodes = shared;
    config.maximum_turns =
        parse_positive_int(positional[2], "maximum turns");
  } else if (positional.size() >= 4U) {
    config.candidate_work =
        parse_unsigned<std::uint64_t>(positional[1], "candidate work");
    config.reference_nodes =
        parse_unsigned<std::uint64_t>(positional[2], "reference nodes");
    config.maximum_turns =
        parse_positive_int(positional[3], "maximum turns");
    if (positional.size() >= 5U) {
      config.batch_start = parse_unsigned<std::uint64_t>(
          positional[4], "batch start", true);
    }
    if (positional.size() >= 6U) {
      config.batch_count =
          parse_unsigned<std::uint64_t>(positional[5], "batch count");
    }
  }

  if (config.batch_count - 1U >
      std::numeric_limits<std::uint64_t>::max() - config.batch_start) {
    throw std::invalid_argument("batch range overflows uint64");
  }
  return config;
}

void verify_complete_action(const ps::GameState &before,
                            const ps::GameState &after,
                            std::string_view action) {
  if (action.empty()) {
    throw std::logic_error("engine returned an empty action");
  }
  ps::GameState replayed = before;
  reference::apply_encoded_turn(replayed, action);
  if (replayed.ball != after.ball || replayed.to_move != after.to_move ||
      replayed.status != after.status ||
      replayed.used_segments != after.used_segments ||
      replayed.visit_count != after.visit_count) {
    throw std::logic_error("engine action did not reproduce its state");
  }
  if (!ps::is_terminal(after) && after.to_move == before.to_move) {
    throw std::logic_error("engine returned an incomplete rebound action");
  }
}

template <typename Config>
void set_candidate_work_budget(Config &config, std::uint64_t work) {
  if constexpr (requires { config.max_work = work; }) {
    config.max_work = work;
  } else if constexpr (requires { config.max_nodes = work; }) {
    // Temporary compatibility for a candidate config that still names its
    // deterministic hard-work counter max_nodes.
    config.max_nodes = work;
  } else {
    static_assert(!std::is_same_v<Config, Config>,
                  "candidate SearchConfig needs max_work or max_nodes");
  }
}

template <typename Config>
void configure_candidate_search(Config &config,
                                const HarnessConfig &harness) {
  if constexpr (requires { config.max_actions = std::size_t{}; }) {
    config.max_actions = harness.candidate_max_actions;
  }
  if constexpr (requires { config.nonroot_actions = std::size_t{}; }) {
    config.nonroot_actions = harness.candidate_nonroot_actions;
  }
  if constexpr (requires { config.replay_value_blend_percent = 15; }) {
    config.replay_value_blend_percent = harness.candidate_replay_blend;
  }
  if constexpr (requires { config.teacher_residual_weight_percent = 100; }) {
    config.teacher_residual_weight_percent = harness.candidate_residual_weight;
  }
  if constexpr (requires { config.exploration_constant = 1.0; }) {
    config.exploration_constant = harness.candidate_exploration;
  }
  if constexpr (requires { config.first_play_urgency = 0.0; }) {
    config.first_play_urgency = harness.candidate_fpu;
  }
  if constexpr (requires { config.final_visit_weight = 0.0; }) {
    config.final_visit_weight = harness.candidate_final_visit_weight;
  }
  if constexpr (requires { config.root_only = false; }) {
    config.root_only = harness.candidate_root_only;
  }
}

Decision choose_reference(ps::GameState &state, std::string_view transcript,
                          const HarnessConfig &harness,
                          std::uint32_t time_budget_ms) {
  const ps::GameState before = state;
  const auto response_started = reference::SearchClock::now();
  Decision decision;
  if (!harness.reference_replay_corrections ||
      !reference::try_replay_correction(
          state, player_id(state.to_move), transcript, decision.action)) {
    reference::SearchConfig config;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent = 100;
    config.max_nodes = harness.reference_nodes;
    const auto started = reference::SearchClock::now();
    if (time_budget_ms != 0) {
      config.absolute_deadline =
          started + std::chrono::milliseconds(time_budget_ms);
    }
    reference::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("rank-4 search returned an overlong action");
      }
      decision.action.push_back(
          reference::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    const auto &stats = search.stats();
    decision.reference_nodes = stats.nodes;
    decision.reference_completed_depth = stats.completed_turn_depth;
    decision.reference_attempted_depth = stats.attempted_turn_depth;
    decision.root_score = stats.root_score;
    decision.budget_exhausted = stats.budget_exhausted;
    decision.node_cap_reached =
        stats.budget_exhausted && stats.nodes >= harness.reference_nodes;
    decision.deadline_reached =
        time_budget_ms != 0 && stats.budget_exhausted &&
        !decision.node_cap_reached;
    decision.searched = true;
  }
  verify_complete_action(before, state, decision.action);
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(
          reference::SearchClock::now() - response_started)
          .count();
  return decision;
}

Decision choose_candidate(ps::GameState &state, std::string_view transcript,
                          const HarnessConfig &harness,
                          std::uint32_t time_budget_ms) {
  const ps::GameState before = state;
  const auto response_started = candidate::SearchClock::now();
  Decision decision;
  if (!harness.candidate_replay_corrections ||
      !candidate::try_replay_correction(state, player_id(state.to_move),
                                        transcript, decision.action)) {
    candidate::SearchConfig config;
    set_candidate_work_budget(config, harness.candidate_work);
    configure_candidate_search(config, harness);
    const auto started = candidate::SearchClock::now();
    if (time_budget_ms != 0) {
      config.absolute_deadline =
          started + std::chrono::milliseconds(time_budget_ms);
    }
    candidate::FullTurnBfmSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("candidate search returned an overlong action");
      }
      decision.action.push_back(
          candidate::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    const auto &stats = search.stats();
    decision.work = stats.work;
    decision.tree_nodes = stats.tree_nodes;
    decision.expansions = stats.expansions;
    decision.child_evaluations = stats.child_evaluations;
    decision.completed_actions = stats.completed_actions;
    decision.generator_partial_paths = stats.generator_partial_paths;
    decision.generator_duplicates = stats.generator_duplicates;
    decision.tactical_actions = stats.tactical_actions;
    decision.generator_truncations = stats.generator_truncations;
    decision.max_deque_size = stats.max_deque_size;
    decision.max_action_edges = stats.max_action_edges;
    decision.root_score = stats.root_score;
    decision.budget_exhausted = stats.budget_exhausted;
    decision.deadline_reached = stats.deadline_reached;
    decision.node_cap_reached = stats.node_cap_reached;
    decision.searched = true;
  }
  verify_complete_action(before, state, decision.action);
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(
          candidate::SearchClock::now() - response_started)
          .count();
  return decision;
}

Opening reconstruct(std::string_view transcript, std::string label,
                    std::string batch_tag) {
  Opening opening{ps::make_initial_state(codingame_rules()),
                  std::string(transcript), std::move(label),
                  std::move(batch_tag)};
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    reference::apply_encoded_turn(opening.state,
                                  transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (ps::is_terminal(opening.state)) {
    throw std::logic_error("curated opening is terminal");
  }
  return opening;
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

std::uint64_t mix_batch(std::uint64_t batch) {
  if (batch == 0) {
    return 0;
  }
  std::uint64_t value = batch + kOpeningRetryStride;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t opening_seed(std::uint64_t batch, int opening_turns,
                           int pair) {
  return kOpeningSeedSalt ^ mix_batch(batch) ^
         (static_cast<std::uint64_t>(opening_turns) << 40U) ^
         static_cast<std::uint64_t>(pair + 1);
}

Opening random_opening(std::uint64_t seed, int opening_turns,
                       std::string label, std::string batch_tag) {
  for (std::uint64_t attempt = 0; attempt < 10'000; ++attempt) {
    std::uint64_t random = seed + attempt * kOpeningRetryStride;
    if (random == 0) {
      random = 1;
    }
    Opening opening{ps::make_initial_state(codingame_rules()), "",
                    label, batch_tag, seed, true};
    bool valid = true;
    for (int turn = 0; turn < opening_turns && valid; ++turn) {
      const ps::Player mover = opening.state.to_move;
      std::string action;
      while (!ps::is_terminal(opening.state) &&
             opening.state.to_move == mover) {
        const std::vector<ps::Move> moves = ps::legal_moves(opening.state);
        if (moves.empty()) {
          valid = false;
          break;
        }
        const ps::Move move = moves[next_random(random) % moves.size()];
        action.push_back(
            reference::encode_direction(opening.state.ball, move.to));
        opening.state = ps::apply_move(opening.state, move);
      }
      if (action.empty() || ps::is_terminal(opening.state)) {
        valid = false;
        break;
      }
      append_turn(opening.transcript, action);
    }
    if (valid && !ps::is_terminal(opening.state)) {
      return opening;
    }
  }
  throw std::runtime_error("could not generate a non-terminal opening");
}

void add_candidate(CandidateTotals &totals, const Decision &decision) {
  totals.work += decision.work;
  totals.tree_nodes += decision.tree_nodes;
  totals.expansions += decision.expansions;
  totals.child_evaluations += decision.child_evaluations;
  totals.completed_actions += decision.completed_actions;
  totals.generator_partial_paths += decision.generator_partial_paths;
  totals.generator_duplicates += decision.generator_duplicates;
  totals.tactical_actions += decision.tactical_actions;
  totals.generator_truncations += decision.generator_truncations;
  totals.searches += decision.searched ? 1U : 0U;
  totals.exhaustions += decision.budget_exhausted ? 1U : 0U;
  totals.deadlines += decision.deadline_reached ? 1U : 0U;
  totals.node_caps += decision.node_cap_reached ? 1U : 0U;
  totals.operational_timeouts += decision.operational_timeout ? 1U : 0U;
  totals.milliseconds += decision.milliseconds;
  totals.maximum_tree_nodes =
      std::max(totals.maximum_tree_nodes, decision.tree_nodes);
  totals.maximum_deque_size =
      std::max(totals.maximum_deque_size, decision.max_deque_size);
  totals.maximum_action_edges =
      std::max(totals.maximum_action_edges, decision.max_action_edges);
  double &maximum = decision.first_response
                        ? totals.maximum_first_milliseconds
                        : totals.maximum_later_milliseconds;
  maximum = std::max(maximum, decision.milliseconds);
}

void add_reference(ReferenceTotals &totals, const Decision &decision) {
  totals.nodes += decision.reference_nodes;
  totals.completed_depth += decision.reference_completed_depth;
  totals.attempted_depth += decision.reference_attempted_depth;
  totals.searches += decision.searched ? 1U : 0U;
  totals.exhaustions += decision.budget_exhausted ? 1U : 0U;
  totals.deadlines += decision.deadline_reached ? 1U : 0U;
  totals.node_caps += decision.node_cap_reached ? 1U : 0U;
  totals.operational_timeouts += decision.operational_timeout ? 1U : 0U;
  totals.milliseconds += decision.milliseconds;
  double &maximum = decision.first_response
                        ? totals.maximum_first_milliseconds
                        : totals.maximum_later_milliseconds;
  maximum = std::max(maximum, decision.milliseconds);
}

GameResult play(const Opening &opening, int candidate_player,
                const HarnessConfig &harness) {
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  GameResult result;

  // The transcript constructs a deterministic starting state; neither engine
  // searched those prefix turns. Each engine's first actual invocation in this
  // standalone game therefore receives its first-response budget.
  std::uint32_t candidate_responses = 0;
  std::uint32_t reference_responses = 0;
  while (!ps::is_terminal(state) && result.turns < harness.maximum_turns) {
    const bool candidate_turn = player_id(state.to_move) == candidate_player;
    const std::uint32_t time_budget =
        candidate_turn
            ? (candidate_responses == 0 ? harness.candidate_first_time_ms
                                        : harness.candidate_later_time_ms)
            : (reference_responses == 0 ? harness.reference_first_time_ms
                                        : harness.reference_later_time_ms);
    Decision decision =
        candidate_turn
            ? choose_candidate(state, transcript, harness, time_budget)
            : choose_reference(state, transcript, harness, time_budget);
    decision.first_response =
        candidate_turn ? candidate_responses == 0 : reference_responses == 0;
    const std::uint32_t operational_limit_ms =
        decision.first_response ? kFirstOperationalLimitMs
                                : kLaterOperationalLimitMs;
    decision.operational_timeout =
        time_budget != 0 && decision.milliseconds >= operational_limit_ms;
    append_turn(transcript, decision.action);
    if (candidate_turn) {
      ++candidate_responses;
      result.candidate_last_root_score = decision.root_score;
      add_candidate(result.candidate, decision);
    } else {
      ++reference_responses;
      result.reference_last_root_score = decision.root_score;
      add_reference(result.reference, decision);
    }
    ++result.turns;
  }
  if (const std::optional<ps::Player> winning = ps::winner(state)) {
    result.winner = player_id(*winning);
  }
  return result;
}

void add_totals(CandidateTotals &into, const CandidateTotals &from) {
  into.work += from.work;
  into.tree_nodes += from.tree_nodes;
  into.expansions += from.expansions;
  into.child_evaluations += from.child_evaluations;
  into.completed_actions += from.completed_actions;
  into.generator_partial_paths += from.generator_partial_paths;
  into.generator_duplicates += from.generator_duplicates;
  into.tactical_actions += from.tactical_actions;
  into.generator_truncations += from.generator_truncations;
  into.searches += from.searches;
  into.exhaustions += from.exhaustions;
  into.deadlines += from.deadlines;
  into.node_caps += from.node_caps;
  into.operational_timeouts += from.operational_timeouts;
  into.milliseconds += from.milliseconds;
  into.maximum_tree_nodes =
      std::max(into.maximum_tree_nodes, from.maximum_tree_nodes);
  into.maximum_deque_size =
      std::max(into.maximum_deque_size, from.maximum_deque_size);
  into.maximum_action_edges =
      std::max(into.maximum_action_edges, from.maximum_action_edges);
  into.maximum_first_milliseconds =
      std::max(into.maximum_first_milliseconds,
               from.maximum_first_milliseconds);
  into.maximum_later_milliseconds =
      std::max(into.maximum_later_milliseconds,
               from.maximum_later_milliseconds);
}

void add_totals(ReferenceTotals &into, const ReferenceTotals &from) {
  into.nodes += from.nodes;
  into.completed_depth += from.completed_depth;
  into.attempted_depth += from.attempted_depth;
  into.searches += from.searches;
  into.exhaustions += from.exhaustions;
  into.deadlines += from.deadlines;
  into.node_caps += from.node_caps;
  into.operational_timeouts += from.operational_timeouts;
  into.milliseconds += from.milliseconds;
  into.maximum_first_milliseconds =
      std::max(into.maximum_first_milliseconds,
               from.maximum_first_milliseconds);
  into.maximum_later_milliseconds =
      std::max(into.maximum_later_milliseconds,
               from.maximum_later_milliseconds);
}

void add_game(Summary &summary, const GameResult &game,
              int candidate_player) {
  ++summary.games;
  ++summary.color_games[candidate_player];
  add_totals(summary.candidate, game.candidate);
  add_totals(summary.reference, game.reference);
  if (game.winner < 0) {
    ++summary.unfinished;
    ++summary.color_unfinished[candidate_player];
  } else if (game.winner == candidate_player) {
    ++summary.candidate_wins;
    ++summary.candidate_color_wins[candidate_player];
  } else {
    ++summary.reference_wins;
    ++summary.reference_color_wins[candidate_player];
  }
}

void print_game(std::string_view prefix, const GameResult &game) {
  std::cout << ' ' << prefix << "=" << game.winner << ':' << game.turns
            << ":cw=" << game.candidate.work
            << ":ct=" << game.candidate.tree_nodes
            << ":cx=" << game.candidate.expansions
            << ":ce=" << game.candidate.child_evaluations
            << ":ca=" << game.candidate.completed_actions
            << ":cp=" << game.candidate.generator_partial_paths
            << ":cd=" << game.candidate.generator_duplicates
            << ":cta=" << game.candidate.tactical_actions
            << ":ctr=" << game.candidate.generator_truncations
            << ":cmax_tree=" << game.candidate.maximum_tree_nodes
            << ":cmax_deque=" << game.candidate.maximum_deque_size
            << ":cmax_edges=" << game.candidate.maximum_action_edges
            << ":rn=" << game.reference.nodes
            << ":searches=" << game.candidate.searches << '/'
            << game.reference.searches
            << ":exhaustions=" << game.candidate.exhaustions << '/'
            << game.reference.exhaustions
            << ":deadlines=" << game.candidate.deadlines << '/'
            << game.reference.deadlines
            << ":node_caps=" << game.candidate.node_caps << '/'
            << game.reference.node_caps
            << ":operational_timeouts="
            << game.candidate.operational_timeouts << '/'
            << game.reference.operational_timeouts
            << ":max_ms=" << game.candidate.maximum_first_milliseconds << ','
            << game.candidate.maximum_later_milliseconds << '/'
            << game.reference.maximum_first_milliseconds << ','
            << game.reference.maximum_later_milliseconds
            << ":scores=" << game.candidate_last_root_score << '/'
            << game.reference_last_root_score;
}

void record_pair(const Opening &opening, const HarnessConfig &harness,
                 Summary &batch_summary, Summary &overall_summary) {
  const std::array<GameResult, 2> games{
      play(opening, 0, harness),
      play(opening, 1, harness),
  };
  std::cout << "opening=" << opening.label
            << " batch=" << opening.batch_tag << " seed=";
  if (opening.seeded) {
    std::cout << opening.seed;
  } else {
    std::cout << '-';
  }
  print_game("c0", games[0]);
  print_game("c1", games[1]);
  std::cout << '\n';
  for (int candidate_player = 0; candidate_player < 2; ++candidate_player) {
    add_game(batch_summary, games[candidate_player], candidate_player);
    add_game(overall_summary, games[candidate_player], candidate_player);
  }
}

double average(std::uint64_t total, std::uint64_t count) {
  return count == 0 ? 0.0
                    : static_cast<double>(total) /
                          static_cast<double>(count);
}

void print_summary(std::string_view kind, std::string_view batch,
                   const Summary &summary, const HarnessConfig &harness) {
  std::cout << kind << " batch=" << batch
            << " games=" << summary.games
            << " candidate=" << summary.candidate_wins
            << " reference=" << summary.reference_wins
            << " unfinished=" << summary.unfinished
            << " candidate_color0=" << summary.candidate_color_wins[0]
            << '/' << summary.color_games[0]
            << " candidate_color1=" << summary.candidate_color_wins[1]
            << '/' << summary.color_games[1]
            << " reference_color0=" << summary.reference_color_wins[0]
            << '/' << summary.color_games[0]
            << " reference_color1=" << summary.reference_color_wins[1]
            << '/' << summary.color_games[1]
            << " unfinished_color0=" << summary.color_unfinished[0]
            << '/' << summary.color_games[0]
            << " unfinished_color1=" << summary.color_unfinished[1]
            << '/' << summary.color_games[1]
            << " candidate_work=" << summary.candidate.work
            << " candidate_tree_nodes=" << summary.candidate.tree_nodes
            << " candidate_expansions=" << summary.candidate.expansions
            << " candidate_child_evaluations="
            << summary.candidate.child_evaluations
            << " candidate_completed_actions="
            << summary.candidate.completed_actions
            << " candidate_generator_partial_paths="
            << summary.candidate.generator_partial_paths
            << " candidate_generator_duplicates="
            << summary.candidate.generator_duplicates
            << " candidate_tactical_actions="
            << summary.candidate.tactical_actions
            << " candidate_generator_truncations="
            << summary.candidate.generator_truncations
            << " candidate_max_tree_nodes="
            << summary.candidate.maximum_tree_nodes
            << " candidate_max_deque_size="
            << summary.candidate.maximum_deque_size
            << " candidate_max_action_edges="
            << summary.candidate.maximum_action_edges
            << " reference_nodes=" << summary.reference.nodes
            << " candidate_searches=" << summary.candidate.searches
            << " reference_searches=" << summary.reference.searches
            << " candidate_exhaustions=" << summary.candidate.exhaustions
            << " reference_exhaustions=" << summary.reference.exhaustions
            << " candidate_deadlines=" << summary.candidate.deadlines
            << " reference_deadlines=" << summary.reference.deadlines
            << " candidate_node_caps=" << summary.candidate.node_caps
            << " reference_node_caps=" << summary.reference.node_caps
            << " candidate_operational_timeouts="
            << summary.candidate.operational_timeouts
            << " reference_operational_timeouts="
            << summary.reference.operational_timeouts
            << " candidate_ms=" << summary.candidate.milliseconds
            << " reference_ms=" << summary.reference.milliseconds
            << " candidate_max_first_ms="
            << summary.candidate.maximum_first_milliseconds
            << " candidate_max_later_ms="
            << summary.candidate.maximum_later_milliseconds
            << " reference_max_first_ms="
            << summary.reference.maximum_first_milliseconds
            << " reference_max_later_ms="
            << summary.reference.maximum_later_milliseconds
            << " reference_completed_depth="
            << average(summary.reference.completed_depth,
                       summary.reference.searches)
            << " reference_attempted_depth="
            << average(summary.reference.attempted_depth,
                       summary.reference.searches)
            << " candidate_work_budget=" << harness.candidate_work
            << " reference_node_budget=" << harness.reference_nodes
            << " candidate_time=" << harness.candidate_first_time_ms << '/'
            << harness.candidate_later_time_ms
            << " reference_time=" << harness.reference_first_time_ms << '/'
            << harness.reference_later_time_ms
            << " operational_limits=" << kFirstOperationalLimitMs << '/'
            << kLaterOperationalLimitMs
            << " candidate_replay=" << harness.candidate_replay_corrections
            << " reference_replay=" << harness.reference_replay_corrections
            << '\n';
}

void compare_position(std::string_view transcript,
                      const HarnessConfig &harness) {
  const Opening opening =
      reconstruct(transcript, "command-line", "position");
  ps::GameState reference_state = opening.state;
  ps::GameState candidate_state = opening.state;
  const std::uint32_t candidate_time =
      harness.position_time_ms != 0 ? harness.position_time_ms
                                    : harness.candidate_first_time_ms;
  const std::uint32_t reference_time =
      harness.position_time_ms != 0 ? harness.position_time_ms
                                    : harness.reference_first_time_ms;
  Decision incumbent =
      choose_reference(reference_state, transcript, harness, reference_time);
  Decision proposed =
      choose_candidate(candidate_state, transcript, harness, candidate_time);
  incumbent.first_response = true;
  proposed.first_response = true;
  incumbent.operational_timeout =
      reference_time != 0 &&
      incumbent.milliseconds >= kFirstOperationalLimitMs;
  proposed.operational_timeout =
      candidate_time != 0 && proposed.milliseconds >= kFirstOperationalLimitMs;
  std::cout << "position=command-line ball=" << opening.state.ball.x << ','
            << opening.state.ball.y
            << " mover=" << player_id(opening.state.to_move)
            << " reference=rank_4:" << incumbent.action
            << ":depth=" << incumbent.reference_completed_depth << '/'
            << incumbent.reference_attempted_depth
            << ":nodes=" << incumbent.reference_nodes
            << ":score=" << incumbent.root_score
            << ":ms=" << incumbent.milliseconds
            << ":exhausted=" << incumbent.budget_exhausted
            << ":deadline=" << incumbent.deadline_reached
            << ":node_cap=" << incumbent.node_cap_reached
            << ":operational_timeout=" << incumbent.operational_timeout
            << " candidate=" << proposed.action
            << ":work=" << proposed.work
            << ":tree=" << proposed.tree_nodes
            << ":expansions=" << proposed.expansions
            << ":evaluations=" << proposed.child_evaluations
            << ":actions=" << proposed.completed_actions
            << ":partials=" << proposed.generator_partial_paths
            << ":duplicates=" << proposed.generator_duplicates
            << ":tactical=" << proposed.tactical_actions
            << ":truncations=" << proposed.generator_truncations
            << ":max_deque=" << proposed.max_deque_size
            << ":max_action_edges=" << proposed.max_action_edges
            << ":score=" << proposed.root_score
            << ":ms=" << proposed.milliseconds
            << ":exhausted=" << proposed.budget_exhausted
            << ":deadline=" << proposed.deadline_reached
            << ":node_cap=" << proposed.node_cap_reached
            << ":operational_timeout=" << proposed.operational_timeout
            << " changed=" << (incumbent.action != proposed.action) << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const HarnessConfig harness = parse_options(argc, argv);
    if (harness.position.has_value()) {
      compare_position(*harness.position, harness);
      return 0;
    }

    Summary overall;
    Summary curated_summary;
    constexpr std::array<std::pair<std::string_view, std::string_view>, 5>
        kCurated{{
            {"0/0", "family-0-0"},
            {"0/0/1", "family-0-0-1"},
            {"0/0/3", "family-0-0-3"},
            {"0/6/5", "family-0-6-5"},
            {"0/6/5/4", "family-0-6-5-4"},
        }};
    for (const auto &[transcript, label] : kCurated) {
      record_pair(reconstruct(transcript, std::string(label), "curated"),
                  harness, curated_summary, overall);
    }
    print_summary("batch_summary", "curated", curated_summary, harness);

    for (std::uint64_t offset = 0; offset < harness.batch_count; ++offset) {
      const std::uint64_t batch = harness.batch_start + offset;
      const std::string batch_tag = "seed-" + std::to_string(batch);
      Summary batch_summary;
      for (const int turns : harness.opening_turns) {
        for (int pair = 0; pair < harness.pairs_per_depth; ++pair) {
          const std::uint64_t seed = opening_seed(batch, turns, pair);
          const std::string label =
              batch_tag + "-random-" + std::to_string(turns) + '-' +
              std::to_string(pair);
          record_pair(random_opening(seed, turns, label, batch_tag), harness,
                      batch_summary, overall);
        }
      }
      print_summary("batch_summary", batch_tag, batch_summary, harness);
    }

    print_summary("summary", "all", overall, harness);
    std::cout << "configuration pairs_per_depth=" << harness.pairs_per_depth
              << " batch_start=" << harness.batch_start
              << " batch_count=" << harness.batch_count
              << " maximum_turns=" << harness.maximum_turns
              << " opening_turns=";
    for (std::size_t index = 0; index < harness.opening_turns.size(); ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      std::cout << harness.opening_turns[index];
    }
    std::cout << " candidate_exploration=" << harness.candidate_exploration
              << " candidate_max_actions=" << harness.candidate_max_actions
              << " candidate_nonroot_actions="
              << harness.candidate_nonroot_actions
              << " candidate_fpu=" << harness.candidate_fpu
              << " candidate_final_visit_weight="
              << harness.candidate_final_visit_weight
              << " candidate_replay_blend="
              << harness.candidate_replay_blend
              << " candidate_residual_weight="
              << harness.candidate_residual_weight
              << " candidate_root_only=" << harness.candidate_root_only
              << " opening_seed_salt=" << kOpeningSeedSalt << '\n';
    return overall.unfinished == 0 &&
                   overall.candidate.operational_timeouts == 0 &&
                   overall.reference.operational_timeouts == 0
               ? 0
               : 2;
  } catch (const std::invalid_argument &error) {
    std::cerr << "rank-4 full-turn BFM comparison gate: " << error.what()
              << '\n';
    return 64;
  } catch (const std::exception &error) {
    std::cerr << "rank-4 full-turn BFM comparison gate: " << error.what()
              << '\n';
    return 70;
  }
}
