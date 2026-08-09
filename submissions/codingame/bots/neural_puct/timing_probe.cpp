#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "submission.cpp"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace candidate = papersoccer::neural_puct;

namespace {

struct Options {
  std::uint32_t first_time_ms{candidate::kFirstSearchTimeMs};
  std::uint32_t later_time_ms{candidate::kLaterSearchTimeMs};
  std::size_t max_nodes{candidate::kMaximumNodes};
  std::uint64_t max_simulations{1'000'000};
  float c_puct{candidate::kDefaultCpuct};
};

struct Measurement {
  std::string label;
  std::uint32_t budget_ms{};
  double milliseconds{};
  std::uint64_t simulations{};
  std::uint64_t nodes{};
  std::uint64_t neural_evaluations{};
  std::uint32_t depth{};
  bool deadline_reached{};
  bool node_cap_reached{};
  std::size_t action_edges{};
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

template <typename Integer>
Integer parse_integer(std::string_view raw, std::string_view label) {
  Integer value{};
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (raw.empty() || error != std::errc{} || position != end || value == 0) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return value;
}

float parse_float(std::string_view raw, std::string_view label) {
  std::string copy(raw);
  char *end = nullptr;
  const float value = std::strtof(copy.c_str(), &end);
  if (copy.empty() || end != copy.c_str() + copy.size() || !(value > 0.0F)) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return value;
}

void apply_transcript(ps::GameState &state, std::string_view transcript) {
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    candidate::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) break;
    begin = separator + 1U;
  }
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("timing position is terminal");
  }
}

ps::GameState position(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  apply_transcript(state, transcript);
  return state;
}

Measurement measure(std::string label, ps::GameState state,
                    std::uint32_t budget_ms, const Options &options) {
  const ps::GameState before = state;
  const auto started = candidate::SearchClock::now();
  candidate::SearchConfig config;
  config.max_nodes = options.max_nodes;
  config.max_simulations = options.max_simulations;
  config.c_puct = options.c_puct;
  config.absolute_deadline =
      started + std::chrono::milliseconds(budget_ms);
  candidate::NeuralPuctSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  const candidate::SearchStats stats = search.stats();

  if (stats.simulations > options.max_simulations) {
    throw std::logic_error("timing probe exceeded the simulation cap");
  }
  if (stats.nodes > options.max_nodes) {
    throw std::logic_error("timing probe exceeded the tree-node cap");
  }

  const ps::Player mover = state.to_move;
  std::string action;
  for (const ps::Move move : moves) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("timing probe returned an overlong action");
    }
    action.push_back(candidate::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (action.empty()) {
    throw std::logic_error("timing probe returned an empty action");
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("timing probe returned an incomplete action");
  }
  ps::GameState replayed = before;
  candidate::apply_encoded_turn(replayed, action);
  if (replayed.ball != state.ball || replayed.to_move != state.to_move ||
      replayed.status != state.status ||
      replayed.used_segments != state.used_segments ||
      replayed.visit_count != state.visit_count) {
    throw std::logic_error("timing probe action did not reproduce its state");
  }

  const double milliseconds =
      std::chrono::duration<double, std::milli>(candidate::SearchClock::now() -
                                                started)
          .count();
  return Measurement{std::move(label),
                     budget_ms,
                     milliseconds,
                     stats.simulations,
                     static_cast<std::uint64_t>(stats.nodes),
                     stats.neural_evaluations,
                     stats.max_depth,
                     stats.deadline_reached,
                     stats.node_cap_reached,
                     moves.size()};
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      std::cout
          << "usage: timing_probe [--first-time-ms N] [--later-time-ms N] "
             "[--max-nodes N] [--max-simulations N] [--c-puct FLOAT]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " +
                                  std::string(argument));
    }
    const std::string value(argv[++index]);
    if (argument == "--first-time-ms") {
      options.first_time_ms =
          parse_integer<std::uint32_t>(value, "first time");
    } else if (argument == "--later-time-ms") {
      options.later_time_ms =
          parse_integer<std::uint32_t>(value, "later time");
    } else if (argument == "--max-nodes") {
      options.max_nodes = parse_integer<std::size_t>(value, "maximum nodes");
    } else if (argument == "--max-simulations") {
      options.max_simulations =
          parse_integer<std::uint64_t>(value, "maximum simulations");
    } else if (argument == "--c-puct") {
      options.c_puct = parse_float(value, "c-puct");
    } else {
      throw std::invalid_argument("unknown argument: " +
                                  std::string(argument));
    }
  }
  return options;
}

void write_measurement(std::ostream &out, const Measurement &measurement) {
  out << "{\"label\":\"" << measurement.label
      << "\",\"budget_ms\":" << measurement.budget_ms
      << ",\"elapsed_ms\":" << measurement.milliseconds
      << ",\"simulations\":" << measurement.simulations
      << ",\"nodes\":" << measurement.nodes
      << ",\"neural_evaluations\":" << measurement.neural_evaluations
      << ",\"max_depth\":" << measurement.depth
      << ",\"deadline_reached\":"
      << (measurement.deadline_reached ? "true" : "false")
      << ",\"node_cap_reached\":"
      << (measurement.node_cap_reached ? "true" : "false")
      << ",\"action_edges\":" << measurement.action_edges << '}';
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    constexpr std::array<std::pair<std::string_view, std::string_view>, 3>
        kShellCases{{
            {"elite-d2",
             "1/7/7/2/1/13/5/74/5027/20555/0/6/61/052434/7/61453/11/"
             "063664/23/6/4/4/2/3"},
            {"rank1-d1",
             "0/0/3/2/7/46160/30/57/6/6/53/61/13/123/556/3/6/143/6/5/0/"
             "143/00/2/55/23"},
            {"elite-d0-dense",
             "5/6/3/3/2/3/50/30675/47/6/3165061/631/671/4/1671/3/"
             "3454707210/3/64121/4/1/3/17/4/4271617/035/075/5/6/3/32107/"
             "743/067"},
        }};

    std::vector<Measurement> measurements;
    measurements.reserve(5);
    measurements.push_back(
        measure("first-player-0", position(""), options.first_time_ms, options));
    measurements.push_back(
        measure("first-player-1", position("0"), options.first_time_ms, options));
    for (const auto &[label, transcript] : kShellCases) {
      measurements.push_back(measure(std::string(label), position(transcript),
                                     options.later_time_ms, options));
    }

    std::cout << std::setprecision(17)
              << "{\"schema\":\"papersoccer.neural-puct-timing.v1\","
              << "\"configuration\":{\"construction_inclusive\":true,"
              << "\"first_budget_ms\":" << options.first_time_ms
              << ",\"later_budget_ms\":" << options.later_time_ms
              << ",\"simulation_cap\":" << options.max_simulations
              << ",\"node_cap\":" << options.max_nodes
              << ",\"c_puct\":" << options.c_puct
              << "},\"measurements\":[";
    for (std::size_t index = 0; index < measurements.size(); ++index) {
      if (index != 0) std::cout << ',';
      write_measurement(std::cout, measurements[index]);
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "neural PUCT timing probe: " << error.what() << '\n';
    return 70;
  }
}
