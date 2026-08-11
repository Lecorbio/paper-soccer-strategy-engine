#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
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

namespace ps = papersoccer;
namespace native = papersoccer::jacek_native_bfm;

namespace {

constexpr std::string_view kRuntimeSchema =
    "papersoccer.jacek-native-runtime-model/v1";
constexpr std::uint32_t kFirstHeadroomLimitMs = 900;
constexpr std::uint32_t kLaterHeadroomLimitMs = 180;
constexpr std::uint32_t kFirstOperationalLimitMs = 1'000;
constexpr std::uint32_t kLaterOperationalLimitMs = 200;

struct Arguments {
  std::string candidate_checkpoint;
  std::string baseline_checkpoint;
  std::size_t pairs{1};
  std::uint32_t first_ms{50};
  std::uint32_t later_ms{10};
  std::size_t maximum_turns{384};
  std::vector<std::size_t> opening_turns{0, 4, 8, 12};
  std::uint64_t seed{0x5a73c1d9e2048b6fULL};
  std::size_t minimum_candidate_wins{};
  std::size_t minimum_wins_per_color{};
  bool verify_only{};
  bool vary_shuffle_seed{};
};

struct LoadedModel {
  std::unique_ptr<native::QuantizedModel> model;
  std::string runtime_sha256;
  std::string model_sha256;
  std::string packed_sha256;
};

struct Opening {
  ps::GameState state;
  std::array<std::uint32_t, 2> player_turns{};
  std::uint64_t seed{};
  std::size_t complete_turns{};
};

struct SearchTotals {
  std::uint64_t decisions{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t completed_actions{};
  std::uint64_t partial_paths{};
  std::uint64_t deadline_searches{};
  std::uint64_t headroom_failures{};
  std::uint64_t operational_timeouts{};
  std::size_t maximum_tree_nodes{};
  double milliseconds{};
  double maximum_first_milliseconds{};
  double maximum_later_milliseconds{};

  void observe(const native::SearchStats &stats, double elapsed,
               std::uint32_t prior_turns) noexcept {
    ++decisions;
    expansions += stats.expansions;
    child_evaluations += stats.child_evaluations;
    completed_actions += stats.completed_actions;
    partial_paths += stats.generator_partial_paths;
    deadline_searches += stats.deadline_reached ? 1U : 0U;
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

std::string procedural_turn(ps::GameState &state, std::uint64_t &random) {
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("cannot extend a terminal procedural opening");
  }
  const ps::Player mover = state.to_move;
  std::string result;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    if (legal.empty()) {
      throw std::logic_error("in-progress opening state has no legal move");
    }
    const ps::Move move = legal[random_index(random, legal.size())];
    result.push_back(native::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
    if (result.size() > 316U) {
      throw std::logic_error("procedural turn exceeded the board edge count");
    }
  }
  return result;
}

Opening make_opening(std::uint64_t base_seed, std::size_t complete_turns) {
  for (std::uint64_t retry = 0; retry < 4'096; ++retry) {
    Opening result{ps::make_initial_state(codingame_rules()), {},
                   base_seed + retry * 0xd1b54a32d192ed03ULL,
                   complete_turns};
    std::uint64_t random = result.seed;
    bool valid = true;
    for (std::size_t turn = 0; turn < complete_turns; ++turn) {
      if (ps::is_terminal(result.state)) {
        valid = false;
        break;
      }
      const int mover = player_id(result.state.to_move);
      (void)procedural_turn(result.state, random);
      ++result.player_turns[static_cast<std::size_t>(mover)];
    }
    if (valid && !ps::is_terminal(result.state)) {
      return result;
    }
  }
  throw std::runtime_error("could not generate a nonterminal opening");
}

std::string choose_turn(ps::GameState &state,
                        const native::QuantizedModel &model,
                        const Arguments &arguments, std::uint32_t prior_turns,
                        std::uint64_t shuffle_seed, SearchTotals &totals) {
  native::SearchConfig config;
  config.max_actions = native::kMaximumActions;
  config.max_partial_paths = native::kProductionPartialPaths;
  config.max_tree_nodes = native::kProductionTreeNodes;
  config.max_expansions = native::kMaximumExpansions;
  config.shuffle_seed = shuffle_seed;
  config.exploration_constant = native::kExplorationConstant;
  config.first_play_urgency = native::kFirstPlayUrgency;
  config.model = &model;
  const std::uint32_t budget =
      prior_turns == 0 ? arguments.first_ms : arguments.later_ms;
  const auto started = native::SearchClock::now();
  config.absolute_deadline = started + std::chrono::milliseconds(budget);
  const native::SearchResult search = native::choose_complete_turn(
      static_cast<const ps::GameState &>(state), config);
  if (search.encoded.empty()) {
    throw std::logic_error("native model returned an empty complete turn");
  }
  ps::GameState next = state;
  native::apply_encoded_turn(next, search.encoded);
  const auto finished = native::SearchClock::now();
  const double elapsed =
      std::chrono::duration<double, std::milli>(finished - started).count();
  totals.observe(search.stats, elapsed, prior_turns);
  state = std::move(next);
  return search.encoded;
}

int play(const Opening &opening, int candidate_player,
         const LoadedModel &candidate, const LoadedModel &baseline,
         const Arguments &arguments, Summary &summary) {
  ps::GameState state = opening.state;
  std::array<std::uint32_t, 2> player_turns = opening.player_turns;
  std::size_t turn = 0;
  for (; turn < arguments.maximum_turns && !ps::is_terminal(state); ++turn) {
    const int mover = player_id(state.to_move);
    const bool candidate_turn = mover == candidate_player;
    const LoadedModel &selected = candidate_turn ? candidate : baseline;
    SearchTotals &totals =
        candidate_turn ? summary.candidate : summary.baseline;
    const std::uint64_t shuffle_seed = arguments.vary_shuffle_seed
        ? arguments.seed ^ opening.seed ^
              (static_cast<std::uint64_t>(turn + 1) *
               0x9e3779b97f4a7c15ULL) ^
              (static_cast<std::uint64_t>(mover) * 0xd1b54a32d192ed03ULL)
        : native::SearchConfig{}.shuffle_seed;
    (void)choose_turn(state, *selected.model, arguments,
                      player_turns[static_cast<std::size_t>(mover)],
                      shuffle_seed, totals);
    ++player_turns[static_cast<std::size_t>(mover)];
  }
  if (const auto winner = ps::winner(state); winner.has_value()) {
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
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

void print_help() {
  std::cout
      << "Usage: jacek_native_model_gate [options]\n"
      << "  --candidate-checkpoint PATH   exported runtime checkpoint\n"
      << "  --baseline-checkpoint PATH    prior runtime checkpoint\n"
      << "  --pairs N                     paired openings; two colors each\n"
      << "  --first-ms N --later-ms N     actual search clocks\n"
      << "  --maximum-turns N             complete turns after the opening\n"
      << "  --opening-turns A,B,...       procedural depths, cycled by pair\n"
      << "  --seed N                      deterministic opening seed\n"
      << "  --minimum-candidate-wins N    total promotion threshold\n"
      << "  --minimum-wins-per-color N    threshold in each candidate color\n"
      << "  --vary-shuffle-seed           diagnostic per-turn shuffle seeds\n"
      << "  --verify-only                 validate and identify both files\n";
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
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
    if (option == "--vary-shuffle-seed") {
      result.vary_shuffle_seed = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument(std::string(option) + " needs a value");
    }
    const std::string_view value = argv[++index];
    if (option == "--candidate-checkpoint") {
      result.candidate_checkpoint = value;
    } else if (option == "--baseline-checkpoint") {
      result.baseline_checkpoint = value;
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
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (result.candidate_checkpoint.empty() ||
      result.baseline_checkpoint.empty()) {
    throw std::invalid_argument("both runtime checkpoints are required");
  }
  if (result.pairs == 0 || result.first_ms == 0 || result.later_ms == 0 ||
      result.first_ms > native::kFirstSearchTimeMs ||
      result.later_ms > native::kLaterSearchTimeMs ||
      result.maximum_turns == 0 || result.opening_turns.empty() ||
      result.minimum_candidate_wins > result.pairs * 2U ||
      result.minimum_wins_per_color > result.pairs) {
    throw std::invalid_argument("model-gate limits are outside supported ranges");
  }
  return result;
}

void print_identity(std::string_view name, const LoadedModel &model) {
  std::cout << name << "_runtime_sha256=" << model.runtime_sha256 << ' '
            << name << "_model_sha256=" << model.model_sha256 << ' '
            << name << "_packed_sha256=" << model.packed_sha256 << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    const LoadedModel candidate =
        load_checkpoint(arguments.candidate_checkpoint);
    const LoadedModel baseline = load_checkpoint(arguments.baseline_checkpoint);
    print_identity("candidate", candidate);
    print_identity("baseline", baseline);
    if (arguments.verify_only) {
      return 0;
    }
    if ((arguments.minimum_candidate_wins != 0 ||
         arguments.minimum_wins_per_color != 0) &&
        candidate.packed_sha256 == baseline.packed_sha256) {
      throw std::invalid_argument(
          "a promotion gate requires two distinct packed checkpoints");
    }

    Summary summary;
    for (std::size_t pair = 0; pair < arguments.pairs; ++pair) {
      const std::size_t depth =
          arguments.opening_turns[pair % arguments.opening_turns.size()];
      const std::uint64_t seed =
          arguments.seed + (pair + 1U) * 0x9e3779b97f4a7c15ULL;
      const Opening opening = make_opening(seed, depth);
      std::cout << "pair=" << pair << " opening_turns=" << depth
                << " seed=" << opening.seed;
      for (int candidate_player = 0; candidate_player < 2;
           ++candidate_player) {
        const int winner = play(opening, candidate_player, candidate, baseline,
                                arguments, summary);
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
        << " candidate_max_tree=" << summary.candidate.maximum_tree_nodes
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
        << " baseline_decisions=" << summary.baseline.decisions
        << " baseline_expansions=" << summary.baseline.expansions
        << " baseline_max_first_ms="
        << summary.baseline.maximum_first_milliseconds
        << " baseline_max_later_ms="
        << summary.baseline.maximum_later_milliseconds
        << " baseline_headroom_failures="
        << summary.baseline.headroom_failures
        << " baseline_operational_timeouts="
        << summary.baseline.operational_timeouts
        << " profile=" << arguments.first_ms << '/' << arguments.later_ms
        << " shuffle_seed_policy="
        << (arguments.vary_shuffle_seed ? "varied" : "deployment-constant")
        << " required_total=" << arguments.minimum_candidate_wins
        << " required_per_color=" << arguments.minimum_wins_per_color
        << " passed=" << (passed ? "true" : "false") << '\n';
    return passed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "jacek_native_model_gate failure: " << error.what() << '\n';
    return 1;
  }
}
