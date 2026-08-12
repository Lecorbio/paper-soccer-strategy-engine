#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include "../submissions/codingame/bots/jacek_native_bfm/bot.cpp"

namespace papersoccer::jacek_native_search_ab {

namespace native = papersoccer::jacek_native_bfm;

inline constexpr std::string_view kRuntimeSchema =
    "papersoccer.jacek-native-runtime-model/v1";
inline constexpr std::uint32_t kFirstHeadroomLimitMs = 900;
inline constexpr std::uint32_t kLaterHeadroomLimitMs = 180;
inline constexpr std::uint32_t kFirstOperationalLimitMs = 1'000;
inline constexpr std::uint32_t kLaterOperationalLimitMs = 200;
inline constexpr std::string_view kTimingScope = "search-through-apply";

enum class FinalFormula {
  Production,
  ValueOnly,
  ValueLogVisitsPlus3,
  ValueLogSelectionVisitsPlus3,
};

struct SearchProfile {
  std::size_t tree_nodes{native::kProductionTreeNodes};
  std::size_t opening_tree_nodes{native::kProductionTreeNodes};
  std::size_t opening_own_decisions{};
  double exploration{native::kExplorationConstant};
  double fpu{native::kFirstPlayUrgency};
  FinalFormula final_formula{FinalFormula::Production};
};

std::size_t tree_nodes_for_own_decision(const SearchProfile &profile,
                                        std::uint32_t prior_turns) noexcept {
  return prior_turns < profile.opening_own_decisions
             ? profile.opening_tree_nodes
             : profile.tree_nodes;
}

struct Arguments {
  std::string checkpoint;
  std::size_t pairs{1};
  std::uint32_t first_ms{50};
  std::uint32_t later_ms{10};
  std::size_t maximum_turns{384};
  std::vector<std::size_t> opening_turns{0, 4, 8, 12};
  std::uint64_t seed{0x5a73c1d9e2048b6fULL};
  std::size_t minimum_candidate_wins{};
  std::size_t minimum_wins_per_color{};
  SearchProfile candidate;
  SearchProfile baseline;
  bool verify_only{};
};

struct ProfileOptions {
  bool legacy_tree_nodes{};
  bool opening_tree_nodes{};
  bool later_tree_nodes{};
  bool opening_own_decisions{};
};

struct LoadedModel {
  std::unique_ptr<native::QuantizedModel> model;
  std::string runtime_sha256;
  std::string model_sha256;
  std::string packed_sha256;
};

struct Opening {
  GameState state;
  std::array<std::uint32_t, 2> player_turns{};
  std::uint64_t seed{};
  std::size_t complete_turns{};
};

struct PhaseTotals {
  std::uint64_t decisions{};
  std::uint64_t deadline_searches{};
  std::uint64_t tree_cap_searches{};
  double milliseconds{};
  double maximum_milliseconds{};
  std::size_t maximum_tree_nodes{};

  void observe(const native::SearchResult &search, double elapsed) noexcept {
    ++decisions;
    deadline_searches += search.stats.deadline_reached ? 1U : 0U;
    tree_cap_searches += search.stats.tree_cap_reached ? 1U : 0U;
    milliseconds += elapsed;
    maximum_milliseconds = std::max(maximum_milliseconds, elapsed);
    maximum_tree_nodes =
        std::max(maximum_tree_nodes, search.stats.tree_nodes);
  }
};

struct SearchTotals {
  std::uint64_t decisions{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t completed_actions{};
  std::uint64_t partial_paths{};
  std::uint64_t deadline_searches{};
  std::uint64_t tree_cap_searches{};
  std::uint64_t final_overrides{};
  std::uint64_t headroom_failures{};
  std::uint64_t operational_timeouts{};
  std::size_t maximum_tree_nodes{};
  double milliseconds{};
  double maximum_first_milliseconds{};
  double maximum_later_milliseconds{};
  PhaseTotals opening;
  PhaseTotals later;

  void observe(const native::SearchResult &search, std::string_view selected,
               double elapsed, std::uint32_t prior_turns,
               std::size_t opening_own_decisions) noexcept {
    const native::SearchStats &stats = search.stats;
    ++decisions;
    expansions += stats.expansions;
    child_evaluations += stats.child_evaluations;
    completed_actions += stats.completed_actions;
    partial_paths += stats.generator_partial_paths;
    deadline_searches += stats.deadline_reached ? 1U : 0U;
    tree_cap_searches += stats.tree_cap_reached ? 1U : 0U;
    final_overrides += selected != search.encoded ? 1U : 0U;
    maximum_tree_nodes = std::max(maximum_tree_nodes, stats.tree_nodes);
    milliseconds += elapsed;
    const bool first = prior_turns == 0;
    if (first) {
      maximum_first_milliseconds =
          std::max(maximum_first_milliseconds, elapsed);
    } else {
      maximum_later_milliseconds =
          std::max(maximum_later_milliseconds, elapsed);
    }
    const std::uint32_t headroom_limit =
        first ? kFirstHeadroomLimitMs : kLaterHeadroomLimitMs;
    const std::uint32_t operational_limit =
        first ? kFirstOperationalLimitMs : kLaterOperationalLimitMs;
    headroom_failures += elapsed >= headroom_limit ? 1U : 0U;
    operational_timeouts += elapsed >= operational_limit ? 1U : 0U;
    (prior_turns < opening_own_decisions ? opening : later)
        .observe(search, elapsed);
  }
};

struct Summary {
  std::size_t candidate_wins{};
  std::size_t baseline_wins{};
  std::size_t candidate_player_one_wins{};
  std::size_t candidate_player_two_wins{};
  std::size_t unfinished{};
  std::size_t games{};
  SearchTotals candidate;
  SearchTotals baseline;
};

bool valid_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

LoadedModel load_checkpoint(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open runtime checkpoint: " + path);
  }
  const std::string raw{std::istreambuf_iterator<char>(input),
                        std::istreambuf_iterator<char>()};
  const std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
  std::istringstream lines(raw);
  std::string schema;
  std::string model_schema;
  std::string feature_schema;
  std::string model_sha;
  std::string packed_sha;
  std::string scales;
  std::string packed;
  std::getline(lines, schema);
  std::getline(lines, model_schema);
  std::getline(lines, feature_schema);
  std::getline(lines, model_sha);
  std::getline(lines, packed_sha);
  std::getline(lines, scales);
  std::getline(lines, packed);
  std::string trailing;
  while (std::getline(lines, trailing)) {
    if (!trailing.empty()) {
      throw std::invalid_argument("runtime checkpoint has trailing data");
    }
  }
  if (schema != kRuntimeSchema || !valid_sha256(model_sha) ||
      !valid_sha256(packed_sha)) {
    throw std::invalid_argument("runtime checkpoint metadata is invalid");
  }
  std::istringstream scale_input(scales);
  float scale_one = 0.0F;
  float scale_two = 0.0F;
  float scale_three = 0.0F;
  if (!(scale_input >> scale_one >> scale_two >> scale_three)) {
    throw std::invalid_argument("runtime checkpoint scales are invalid");
  }
  scale_input >> std::ws;
  if (!scale_input.eof()) {
    throw std::invalid_argument("runtime checkpoint scales have trailing data");
  }
  auto model = std::make_unique<native::QuantizedModel>(
      packed, scale_one, scale_two, scale_three, model_schema, feature_schema,
      model_sha, model_sha, false, packed_sha);
  if (model->model_sha256() != model_sha ||
      model->packed_sha256() != packed_sha) {
    throw std::invalid_argument("runtime checkpoint hashes do not match payload");
  }
  return LoadedModel{
      std::move(model),
      native::sha256_hex(std::span<const std::uint8_t>(bytes)),
      std::move(model_sha),
      std::move(packed_sha),
  };
}

RulesConfig codingame_rules() {
  return RulesConfig{8, 10, GoalRule::OwnGoalsAllowed,
                     BlockedRule::MoverLoses};
}

int player_id(Player player) noexcept {
  return player == Player::One ? 0 : 1;
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

void procedural_turn(GameState &state, std::uint64_t &random) {
  if (is_terminal(state)) {
    throw std::invalid_argument("cannot extend a terminal procedural opening");
  }
  const Player mover = state.to_move;
  std::size_t length = 0;
  while (!is_terminal(state) && state.to_move == mover) {
    const std::vector<Move> legal = legal_moves(state);
    if (legal.empty()) {
      throw std::logic_error("in-progress opening state has no legal move");
    }
    state = apply_move(state, legal[random_index(random, legal.size())]);
    if (++length > native::kEdgeInputs) {
      throw std::logic_error("procedural turn exceeded the board edge count");
    }
  }
}

Opening make_opening(std::uint64_t base_seed, std::size_t complete_turns) {
  for (std::uint64_t retry = 0; retry < 4'096; ++retry) {
    Opening result{make_initial_state(codingame_rules()), {},
                   base_seed + retry * 0xd1b54a32d192ed03ULL,
                   complete_turns};
    std::uint64_t random = result.seed;
    bool valid = true;
    for (std::size_t turn = 0; turn < complete_turns; ++turn) {
      if (is_terminal(result.state)) {
        valid = false;
        break;
      }
      const int mover = player_id(result.state.to_move);
      procedural_turn(result.state, random);
      ++result.player_turns[static_cast<std::size_t>(mover)];
    }
    if (valid && !is_terminal(result.state)) return result;
  }
  throw std::runtime_error("could not generate a nonterminal opening");
}

std::string_view final_formula_name(FinalFormula formula) noexcept {
  switch (formula) {
    case FinalFormula::Production:
      return "value-log-visits";
    case FinalFormula::ValueOnly:
      return "value-only";
    case FinalFormula::ValueLogVisitsPlus3:
      return "value-log-visits-plus3";
    case FinalFormula::ValueLogSelectionVisitsPlus3:
      return "value-log-selection-visits-plus3";
  }
  return "invalid";
}

FinalFormula parse_final_formula(std::string_view value) {
  if (value == "value-log-visits" || value == "production") {
    return FinalFormula::Production;
  }
  if (value == "value-only") return FinalFormula::ValueOnly;
  if (value == "value-log-visits-plus3") {
    return FinalFormula::ValueLogVisitsPlus3;
  }
  if (value == "value-log-selection-visits-plus3") {
    return FinalFormula::ValueLogSelectionVisitsPlus3;
  }
  throw std::invalid_argument(
      "final formula must be value-log-visits, value-only, "
      "value-log-visits-plus3, or value-log-selection-visits-plus3");
}

double final_score(const native::RootActionStat &action,
                   FinalFormula formula) {
  if (!std::isfinite(action.value)) {
    throw std::invalid_argument("root action value is not finite");
  }
  switch (formula) {
    case FinalFormula::Production:
      return native::final_action_score(action.value, action.visits);
    case FinalFormula::ValueOnly:
      return static_cast<double>(action.value);
    case FinalFormula::ValueLogVisitsPlus3:
      return static_cast<double>(action.value) +
             std::log(static_cast<double>(action.visits) + 3.0);
    case FinalFormula::ValueLogSelectionVisitsPlus3:
      return static_cast<double>(action.value) +
             std::log(static_cast<double>(action.selection_visits) + 3.0);
  }
  throw std::logic_error("invalid final formula");
}

std::string select_action(const native::SearchResult &search,
                          FinalFormula formula) {
  if (search.root_actions.empty()) {
    throw std::logic_error("native search returned no root actions");
  }
  const native::RootActionStat *best = &search.root_actions.front();
  double best_score = final_score(*best, formula);
  for (const native::RootActionStat &action : search.root_actions) {
    const double score = final_score(action, formula);
    if (score > best_score ||
        (score == best_score && action.encoded < best->encoded)) {
      best = &action;
      best_score = score;
    }
  }
  return best->encoded;
}

std::string choose_turn(GameState &state, const native::QuantizedModel &model,
                        const SearchProfile &profile,
                        const Arguments &arguments, std::uint32_t prior_turns,
                        SearchTotals &totals) {
  native::SearchConfig config;
  config.max_actions = native::kMaximumActions;
  config.max_partial_paths = native::kProductionPartialPaths;
  config.max_tree_nodes = tree_nodes_for_own_decision(profile, prior_turns);
  config.max_expansions = native::kMaximumExpansions;
  config.shuffle_seed = native::SearchConfig{}.shuffle_seed;
  config.exploration_constant = profile.exploration;
  config.first_play_urgency = profile.fpu;
  config.model = &model;
  const std::uint32_t budget =
      prior_turns == 0 ? arguments.first_ms : arguments.later_ms;
  const auto started = native::SearchClock::now();
  config.absolute_deadline = started + std::chrono::milliseconds(budget);
  const native::SearchResult search = native::choose_complete_turn(
      static_cast<const GameState &>(state), config);
  const std::string selected = select_action(search, profile.final_formula);
  GameState next = state;
  native::apply_encoded_turn(next, selected);
  const auto finished = native::SearchClock::now();
  const double elapsed =
      std::chrono::duration<double, std::milli>(finished - started).count();
  totals.observe(search, selected, elapsed, prior_turns,
                 profile.opening_own_decisions);
  state = std::move(next);
  return selected;
}

int play(const Opening &opening, int candidate_player,
         const LoadedModel &runtime, const Arguments &arguments,
         Summary &summary) {
  GameState state = opening.state;
  std::array<std::uint32_t, 2> player_turns = opening.player_turns;
  for (std::size_t turn = 0;
       turn < arguments.maximum_turns && !is_terminal(state); ++turn) {
    const int mover = player_id(state.to_move);
    const bool candidate_turn = mover == candidate_player;
    const SearchProfile &profile =
        candidate_turn ? arguments.candidate : arguments.baseline;
    SearchTotals &totals =
        candidate_turn ? summary.candidate : summary.baseline;
    (void)choose_turn(state, *runtime.model, profile, arguments,
                      player_turns[static_cast<std::size_t>(mover)], totals);
    ++player_turns[static_cast<std::size_t>(mover)];
  }
  if (const auto winner = papersoccer::winner(state); winner.has_value()) {
    return player_id(*winner);
  }
  return -1;
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

double parse_double(std::string_view text, std::string_view label) {
  double value{};
  const char *first = text.data();
  const char *last = first + text.size();
  const auto [end, error] =
      std::from_chars(first, last, value, std::chars_format::general);
  if (error != std::errc{} || end != last || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(label) + " must be finite");
  }
  return value;
}

std::vector<std::size_t> parse_opening_turns(std::string_view text) {
  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end =
        comma == std::string_view::npos ? text.size() : comma;
    if (start == end) {
      throw std::invalid_argument("opening turns contain an empty value");
    }
    result.push_back(parse_integer<std::size_t>(
        text.substr(start, end - start), "opening turns"));
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return result;
}

std::string opening_turns_text(
    const std::vector<std::size_t> &opening_turns) {
  std::string result;
  for (const std::size_t value : opening_turns) {
    if (!result.empty()) result.push_back(',');
    result.append(std::to_string(value));
  }
  return result;
}

void validate_profile(const SearchProfile &profile, std::string_view name) {
  if (profile.tree_nodes < 2 || profile.opening_tree_nodes < 2 ||
      profile.tree_nodes > native::kMaximumTreeNodes ||
      profile.opening_tree_nodes > native::kMaximumTreeNodes ||
      !std::isfinite(profile.exploration) || profile.exploration < 0.0 ||
      !std::isfinite(profile.fpu)) {
    throw std::invalid_argument(std::string(name) +
                                " search profile is outside native bounds");
  }
}

void print_help() {
  std::cout
      << "Usage: jacek_native_search_ab_gate [options]\n"
      << "  --checkpoint PATH             one runtime used by both profiles\n"
      << "  --pairs N                     paired openings; two colors each\n"
      << "  --first-ms N --later-ms N     shared actual search clocks\n"
      << "  --maximum-turns N             complete turns after the opening\n"
      << "  --opening-turns A,B,...       procedural depths, cycled by pair\n"
      << "  --seed N                      deterministic opening seed\n"
      << "  --candidate-tree-nodes N --baseline-tree-nodes N\n"
      << "  or, per side, all of:\n"
      << "     --SIDE-opening-tree-nodes N --SIDE-later-tree-nodes N\n"
      << "     --SIDE-opening-own-decisions N\n"
      << "  --candidate-c X --baseline-c X\n"
      << "  --candidate-fpu X --baseline-fpu X\n"
      << "  --candidate-final NAME --baseline-final NAME\n"
      << "     NAME: value-log-visits | value-only |\n"
      << "           value-log-visits-plus3 |\n"
      << "           value-log-selection-visits-plus3\n"
      << "  --minimum-candidate-wins N    optional total threshold\n"
      << "  --minimum-wins-per-color N    optional threshold per color\n"
      << "  --verify-only                 identify and validate the runtime\n";
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  ProfileOptions candidate_options;
  ProfileOptions baseline_options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--help") {
      print_help();
      std::exit(0);
    }
    if (option == "--verify-only") {
      result.verify_only = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(option) + " needs a value");
    }
    const std::string_view value = argv[++index];
    if (option == "--checkpoint") {
      result.checkpoint = value;
    } else if (option == "--pairs") {
      result.pairs = parse_integer<std::size_t>(value, "pairs");
    } else if (option == "--first-ms") {
      result.first_ms = parse_integer<std::uint32_t>(value, "first ms");
    } else if (option == "--later-ms") {
      result.later_ms = parse_integer<std::uint32_t>(value, "later ms");
    } else if (option == "--maximum-turns") {
      result.maximum_turns =
          parse_integer<std::size_t>(value, "maximum turns");
    } else if (option == "--opening-turns") {
      result.opening_turns = parse_opening_turns(value);
    } else if (option == "--seed") {
      result.seed = parse_integer<std::uint64_t>(value, "seed");
    } else if (option == "--minimum-candidate-wins") {
      result.minimum_candidate_wins =
          parse_integer<std::size_t>(value, "minimum candidate wins");
    } else if (option == "--minimum-wins-per-color") {
      result.minimum_wins_per_color =
          parse_integer<std::size_t>(value, "minimum wins per color");
    } else if (option == "--candidate-tree-nodes") {
      result.candidate.tree_nodes =
          parse_integer<std::size_t>(value, "candidate tree nodes");
      candidate_options.legacy_tree_nodes = true;
    } else if (option == "--baseline-tree-nodes") {
      result.baseline.tree_nodes =
          parse_integer<std::size_t>(value, "baseline tree nodes");
      baseline_options.legacy_tree_nodes = true;
    } else if (option == "--candidate-opening-tree-nodes") {
      result.candidate.opening_tree_nodes = parse_integer<std::size_t>(
          value, "candidate opening tree nodes");
      candidate_options.opening_tree_nodes = true;
    } else if (option == "--candidate-later-tree-nodes") {
      result.candidate.tree_nodes =
          parse_integer<std::size_t>(value, "candidate later tree nodes");
      candidate_options.later_tree_nodes = true;
    } else if (option == "--candidate-opening-own-decisions") {
      result.candidate.opening_own_decisions = parse_integer<std::size_t>(
          value, "candidate opening own decisions");
      candidate_options.opening_own_decisions = true;
    } else if (option == "--baseline-opening-tree-nodes") {
      result.baseline.opening_tree_nodes = parse_integer<std::size_t>(
          value, "baseline opening tree nodes");
      baseline_options.opening_tree_nodes = true;
    } else if (option == "--baseline-later-tree-nodes") {
      result.baseline.tree_nodes =
          parse_integer<std::size_t>(value, "baseline later tree nodes");
      baseline_options.later_tree_nodes = true;
    } else if (option == "--baseline-opening-own-decisions") {
      result.baseline.opening_own_decisions = parse_integer<std::size_t>(
          value, "baseline opening own decisions");
      baseline_options.opening_own_decisions = true;
    } else if (option == "--candidate-c") {
      result.candidate.exploration = parse_double(value, "candidate C");
    } else if (option == "--baseline-c") {
      result.baseline.exploration = parse_double(value, "baseline C");
    } else if (option == "--candidate-fpu") {
      result.candidate.fpu = parse_double(value, "candidate FPU");
    } else if (option == "--baseline-fpu") {
      result.baseline.fpu = parse_double(value, "baseline FPU");
    } else if (option == "--candidate-final") {
      result.candidate.final_formula = parse_final_formula(value);
    } else if (option == "--baseline-final") {
      result.baseline.final_formula = parse_final_formula(value);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (result.checkpoint.empty()) {
    throw std::invalid_argument("one shared runtime checkpoint is required");
  }
  const auto finalize_profile = [](SearchProfile &profile,
                                   const ProfileOptions &options,
                                   std::string_view name) {
    const bool any_split = options.opening_tree_nodes ||
                           options.later_tree_nodes ||
                           options.opening_own_decisions;
    const bool all_split = options.opening_tree_nodes &&
                           options.later_tree_nodes &&
                           options.opening_own_decisions;
    if (options.legacy_tree_nodes && any_split) {
      throw std::invalid_argument(std::string(name) +
                                  " profile mixes legacy and phase caps");
    }
    if (any_split != all_split) {
      throw std::invalid_argument(std::string(name) +
                                  " phase profile must specify all fields");
    }
    if (all_split && profile.opening_own_decisions == 0) {
      throw std::invalid_argument(std::string(name) +
                                  " phase cutoff must be positive");
    }
    if (options.legacy_tree_nodes) {
      profile.opening_tree_nodes = profile.tree_nodes;
      profile.opening_own_decisions = 0;
    }
  };
  finalize_profile(result.candidate, candidate_options, "candidate");
  finalize_profile(result.baseline, baseline_options, "baseline");
  if (result.pairs == 0 || result.first_ms == 0 || result.later_ms == 0 ||
      result.first_ms > native::kFirstSearchTimeMs ||
      result.later_ms > native::kLaterSearchTimeMs ||
      result.maximum_turns == 0 || result.opening_turns.empty() ||
      result.candidate.opening_own_decisions > result.maximum_turns ||
      result.baseline.opening_own_decisions > result.maximum_turns ||
      result.minimum_candidate_wins > result.pairs * 2U ||
      result.minimum_wins_per_color > result.pairs) {
    throw std::invalid_argument("search A/B limits are outside supported ranges");
  }
  validate_profile(result.candidate, "candidate");
  validate_profile(result.baseline, "baseline");
  return result;
}

void print_identity(const LoadedModel &model) {
  std::cout << "runtime_sha256=" << model.runtime_sha256 << ' '
            << "model_sha256=" << model.model_sha256 << ' '
            << "packed_sha256=" << model.packed_sha256 << '\n';
}

void print_profile(std::string_view name, const SearchProfile &profile) {
  std::cout << name << "_opening_tree_nodes="
            << profile.opening_tree_nodes << ' '
            << name << "_later_tree_nodes=" << profile.tree_nodes << ' '
            << name << "_opening_own_decisions="
            << profile.opening_own_decisions << ' '
            << name << "_c=" << profile.exploration << ' '
            << name << "_fpu=" << profile.fpu << ' '
            << name << "_final=" << final_formula_name(profile.final_formula)
            << '\n';
}

std::array<int, 2> candidate_player_order(std::size_t pair) noexcept {
  return pair % 2U == 0U ? std::array<int, 2>{0, 1}
                         : std::array<int, 2>{1, 0};
}

int main_impl(int argc, char **argv) {
  const Arguments arguments = parse_arguments(argc, argv);
  const LoadedModel runtime = load_checkpoint(arguments.checkpoint);
  print_identity(runtime);
  print_profile("candidate", arguments.candidate);
  print_profile("baseline", arguments.baseline);
  if (arguments.verify_only) return 0;

  Summary summary;
  for (std::size_t pair = 0; pair < arguments.pairs; ++pair) {
    const std::size_t depth =
        arguments.opening_turns[pair % arguments.opening_turns.size()];
    const std::uint64_t seed =
        arguments.seed + (pair + 1U) * 0x9e3779b97f4a7c15ULL;
    const Opening opening = make_opening(seed, depth);
    std::cout << "pair=" << pair << " opening_turns=" << depth
              << " seed=" << opening.seed;
    for (const int candidate_player : candidate_player_order(pair)) {
      const int winner =
          play(opening, candidate_player, runtime, arguments, summary);
      ++summary.games;
      if (winner < 0) {
        ++summary.unfinished;
      } else if (winner == candidate_player) {
        ++summary.candidate_wins;
        if (candidate_player == 0) {
          ++summary.candidate_player_one_wins;
        } else {
          ++summary.candidate_player_two_wins;
        }
      } else {
        ++summary.baseline_wins;
      }
      std::cout << " c" << candidate_player << '=' << winner;
    }
    std::cout << '\n';
  }

  const bool passed =
      summary.unfinished == 0 &&
      summary.candidate.headroom_failures == 0 &&
      summary.baseline.headroom_failures == 0 &&
      summary.candidate.operational_timeouts == 0 &&
      summary.baseline.operational_timeouts == 0 &&
      summary.candidate_wins >= arguments.minimum_candidate_wins &&
      summary.candidate_player_one_wins >=
          arguments.minimum_wins_per_color &&
      summary.candidate_player_two_wins >=
          arguments.minimum_wins_per_color;
  std::cout
      << "summary candidate=" << summary.candidate_wins
      << " baseline=" << summary.baseline_wins
      << " unfinished=" << summary.unfinished
      << " candidate_player_one=" << summary.candidate_player_one_wins
      << " candidate_player_two=" << summary.candidate_player_two_wins
      << " games=" << summary.games
      << " candidate_decisions=" << summary.candidate.decisions
      << " candidate_expansions=" << summary.candidate.expansions
      << " candidate_child_evaluations="
      << summary.candidate.child_evaluations
      << " candidate_completed_actions="
      << summary.candidate.completed_actions
      << " candidate_partial_paths=" << summary.candidate.partial_paths
      << " candidate_max_tree=" << summary.candidate.maximum_tree_nodes
      << " candidate_tree_cap_searches="
      << summary.candidate.tree_cap_searches
      << " candidate_final_overrides=" << summary.candidate.final_overrides
      << " candidate_ms=" << summary.candidate.milliseconds
      << " candidate_max_first_ms="
      << summary.candidate.maximum_first_milliseconds
      << " candidate_max_later_ms="
      << summary.candidate.maximum_later_milliseconds
      << " candidate_deadline_searches="
      << summary.candidate.deadline_searches
      << " candidate_headroom_failures="
      << summary.candidate.headroom_failures
      << " candidate_operational_timeouts="
      << summary.candidate.operational_timeouts
      << " candidate_opening_decisions="
      << summary.candidate.opening.decisions
      << " candidate_opening_deadline_searches="
      << summary.candidate.opening.deadline_searches
      << " candidate_opening_tree_cap_searches="
      << summary.candidate.opening.tree_cap_searches
      << " candidate_opening_ms=" << summary.candidate.opening.milliseconds
      << " candidate_opening_max_ms="
      << summary.candidate.opening.maximum_milliseconds
      << " candidate_opening_max_tree="
      << summary.candidate.opening.maximum_tree_nodes
      << " candidate_later_decisions=" << summary.candidate.later.decisions
      << " candidate_later_deadline_searches="
      << summary.candidate.later.deadline_searches
      << " candidate_later_tree_cap_searches="
      << summary.candidate.later.tree_cap_searches
      << " candidate_later_ms=" << summary.candidate.later.milliseconds
      << " candidate_later_max_ms="
      << summary.candidate.later.maximum_milliseconds
      << " candidate_later_max_tree="
      << summary.candidate.later.maximum_tree_nodes
      << " baseline_decisions=" << summary.baseline.decisions
      << " baseline_expansions=" << summary.baseline.expansions
      << " baseline_child_evaluations="
      << summary.baseline.child_evaluations
      << " baseline_completed_actions="
      << summary.baseline.completed_actions
      << " baseline_partial_paths=" << summary.baseline.partial_paths
      << " baseline_max_tree=" << summary.baseline.maximum_tree_nodes
      << " baseline_tree_cap_searches=" << summary.baseline.tree_cap_searches
      << " baseline_final_overrides=" << summary.baseline.final_overrides
      << " baseline_ms=" << summary.baseline.milliseconds
      << " baseline_max_first_ms="
      << summary.baseline.maximum_first_milliseconds
      << " baseline_max_later_ms="
      << summary.baseline.maximum_later_milliseconds
      << " baseline_deadline_searches=" << summary.baseline.deadline_searches
      << " baseline_headroom_failures="
      << summary.baseline.headroom_failures
      << " baseline_operational_timeouts="
      << summary.baseline.operational_timeouts
      << " baseline_opening_decisions="
      << summary.baseline.opening.decisions
      << " baseline_opening_deadline_searches="
      << summary.baseline.opening.deadline_searches
      << " baseline_opening_tree_cap_searches="
      << summary.baseline.opening.tree_cap_searches
      << " baseline_opening_ms=" << summary.baseline.opening.milliseconds
      << " baseline_opening_max_ms="
      << summary.baseline.opening.maximum_milliseconds
      << " baseline_opening_max_tree="
      << summary.baseline.opening.maximum_tree_nodes
      << " baseline_later_decisions=" << summary.baseline.later.decisions
      << " baseline_later_deadline_searches="
      << summary.baseline.later.deadline_searches
      << " baseline_later_tree_cap_searches="
      << summary.baseline.later.tree_cap_searches
      << " baseline_later_ms=" << summary.baseline.later.milliseconds
      << " baseline_later_max_ms="
      << summary.baseline.later.maximum_milliseconds
      << " baseline_later_max_tree="
      << summary.baseline.later.maximum_tree_nodes
      << " profile=" << arguments.first_ms << '/' << arguments.later_ms
      << " pairs=" << arguments.pairs
      << " maximum_turns=" << arguments.maximum_turns
      << " opening_turns=" << opening_turns_text(arguments.opening_turns)
      << " opening_seed=" << arguments.seed
      << " shuffle_seed_policy=deployment-constant"
      << " runtime_policy=same"
      << " game_order_policy=pair-parity-color-swap"
      << " timing_scope=" << kTimingScope
      << " candidate_opening_tree_nodes="
      << arguments.candidate.opening_tree_nodes
      << " candidate_later_tree_nodes=" << arguments.candidate.tree_nodes
      << " candidate_opening_own_decisions="
      << arguments.candidate.opening_own_decisions
      << " candidate_c=" << arguments.candidate.exploration
      << " candidate_fpu=" << arguments.candidate.fpu
      << " candidate_final="
      << final_formula_name(arguments.candidate.final_formula)
      << " baseline_opening_tree_nodes="
      << arguments.baseline.opening_tree_nodes
      << " baseline_later_tree_nodes=" << arguments.baseline.tree_nodes
      << " baseline_opening_own_decisions="
      << arguments.baseline.opening_own_decisions
      << " baseline_c=" << arguments.baseline.exploration
      << " baseline_fpu=" << arguments.baseline.fpu
      << " baseline_final="
      << final_formula_name(arguments.baseline.final_formula)
      << " required_total=" << arguments.minimum_candidate_wins
      << " required_per_color=" << arguments.minimum_wins_per_color
      << " passed=" << (passed ? "true" : "false") << '\n';
  return passed ? 0 : 1;
}

}  // namespace papersoccer::jacek_native_search_ab

#if !defined(PAPER_SOCCER_JACEK_NATIVE_SEARCH_AB_GATE_NO_MAIN)
int main(int argc, char **argv) {
  try {
    return papersoccer::jacek_native_search_ab::main_impl(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "jacek_native_search_ab_gate failure: " << error.what()
              << '\n';
    return 1;
  }
}
#endif
