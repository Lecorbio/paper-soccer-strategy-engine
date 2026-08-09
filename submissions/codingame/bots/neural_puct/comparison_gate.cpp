#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank5_engine
#include "../rank_5/bot.cpp"
#undef turn_action_v2

#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "bot.cpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

namespace ps = papersoccer;
namespace candidate = papersoccer::neural_puct;
namespace incumbent = papersoccer::rank5_engine;

namespace {

constexpr std::string_view kPromotionBankHeader =
    "opening_id\tsplit\tstratum\tsource_agent_id\tsource_game_id\t"
    "opponent_agent_id\twinner_player_id\tturn_index\tphysical_edges\t"
    "state_key\tcanonical_key\tball_x\tball_y\tmover\twinner_tier\t"
    "goal_distance_band\tused_edge_band\tshell_edge_band\topening_family\t"
    "observed_winner_action\ttranscript";

struct Options {
  std::string bank;
  std::string output{"-"};
  std::uint64_t candidate_simulations{1'000};
  std::size_t candidate_nodes{100'000};
  std::uint64_t incumbent_nodes{5'000};
  std::uint32_t time_budget_ms{};
  int maximum_turns{320};
  std::size_t opening_limit{std::numeric_limits<std::size_t>::max()};
  int shard_count{1};
  int shard_index{};
  float c_puct{candidate::kDefaultCpuct};
  bool candidate_simulations_explicit{};
  bool incumbent_nodes_explicit{};
};

struct Opening {
  std::string id;
  std::string transcript;
  ps::GameState state;
};

struct Decision {
  std::string action;
  std::uint64_t simulations{};
  std::uint64_t nodes{};
  std::uint64_t neural_evaluations{};
  std::uint32_t depth{};
  double milliseconds{};
  bool deadline_reached{};
  bool node_cap_reached{};
};

struct Work {
  std::uint64_t decisions{};
  std::uint64_t searches{};
  std::uint64_t simulations{};
  std::uint64_t nodes{};
  std::uint64_t neural_evaluations{};
  std::uint64_t depth_sum{};
  std::uint64_t deadlines{};
  std::uint64_t node_caps{};
  double milliseconds{};
  double maximum_milliseconds{};
};

struct GameResult {
  std::string opening_id;
  int candidate_player{};
  int winner{-1};
  int turns{};
  bool illegal{};
  std::string illegal_engine;
  std::string error;
  Work candidate_work;
  Work incumbent_work;
};

struct Summary {
  int games{};
  int candidate_wins{};
  int incumbent_wins{};
  int unfinished{};
  int illegal_actions{};
  int candidate_illegal_actions{};
  int incumbent_illegal_actions{};
  std::array<int, 2> color_games{};
  std::array<int, 2> color_wins{};
  Work candidate_work;
  Work incumbent_work;
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
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

std::vector<std::string_view> split_tabs(std::string_view line) {
  std::vector<std::string_view> fields;
  std::size_t begin = 0;
  while (true) {
    const std::size_t separator = line.find('\t', begin);
    if (separator == std::string_view::npos) {
      fields.push_back(line.substr(begin));
      return fields;
    }
    fields.push_back(line.substr(begin, separator - begin));
    begin = separator + 1U;
  }
}

template <typename Integer>
Integer parse_integer(std::string_view raw, std::string_view label,
                      bool allow_zero = false) {
  Integer value{};
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (raw.empty() || error != std::errc{} || position != end ||
      (!allow_zero && value <= 0)) {
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

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) transcript.push_back('/');
  transcript.append(action);
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    if (end == begin) {
      throw std::invalid_argument("opening contains an empty turn");
    }
    incumbent::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) break;
    begin = separator + 1U;
  }
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("opening is terminal");
  }
  return state;
}

Opening make_opening(std::string id, std::string transcript) {
  return Opening{std::move(id), transcript, reconstruct(transcript)};
}

std::vector<Opening> default_openings() {
  constexpr std::array<std::pair<std::string_view, std::string_view>, 6>
      kOpenings{{
          {"initial", ""},
          {"family-0-0", "0/0"},
          {"family-0-0-1", "0/0/1"},
          {"family-0-0-3", "0/0/3"},
          {"family-0-6-5", "0/6/5"},
          {"family-0-6-5-4", "0/6/5/4"},
      }};
  std::vector<Opening> openings;
  openings.reserve(kOpenings.size());
  for (const auto &[id, transcript] : kOpenings) {
    openings.push_back(make_opening(std::string(id), std::string(transcript)));
  }
  return openings;
}

std::vector<Opening> read_bank(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("could not open bank: " + path);

  std::string first;
  if (!std::getline(input, first)) {
    throw std::invalid_argument("opening bank is empty");
  }
  if (!first.empty() && first.back() == '\r') first.pop_back();

  enum class Format { Promotion, Minimal, Plain };
  Format format = Format::Plain;
  if (first == kPromotionBankHeader) {
    format = Format::Promotion;
  } else if (first == "opening_id\ttranscript") {
    format = Format::Minimal;
  }

  std::vector<Opening> openings;
  std::size_t line_number = 1;
  auto add_line = [&](std::string_view line) {
    if (line.empty()) return;
    if (format == Format::Plain) {
      openings.push_back(make_opening(
          "line-" + std::to_string(line_number), std::string(line)));
      return;
    }
    const std::vector<std::string_view> fields = split_tabs(line);
    const std::size_t expected = format == Format::Promotion ? 21U : 2U;
    if (fields.size() != expected || fields[0].empty()) {
      throw std::invalid_argument("invalid opening bank row at line " +
                                  std::to_string(line_number));
    }
    const std::string_view raw_transcript = fields.back();
    openings.push_back(make_opening(
        std::string(fields[0]),
        raw_transcript == "-" ? std::string{} : std::string(raw_transcript)));
  };

  if (format == Format::Plain) add_line(first);
  std::string line;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    add_line(line);
  }
  if (openings.empty()) throw std::invalid_argument("opening bank is empty");
  return openings;
}

void verify_complete_action(const ps::GameState &before,
                            const ps::GameState &after,
                            std::string_view action) {
  if (action.empty()) throw std::logic_error("engine returned an empty action");
  ps::GameState replayed = before;
  incumbent::apply_encoded_turn(replayed, action);
  if (replayed.ball != after.ball || replayed.to_move != after.to_move ||
      replayed.status != after.status ||
      replayed.used_segments != after.used_segments ||
      replayed.visit_count != after.visit_count) {
    throw std::logic_error("encoded action does not reproduce engine state");
  }
  if (!ps::is_terminal(after) && after.to_move == before.to_move) {
    throw std::logic_error("engine returned an incomplete rebound sequence");
  }
}

Decision choose_candidate(ps::GameState &state, const Options &options) {
  const ps::GameState before = state;
  const auto started = candidate::SearchClock::now();
  candidate::SearchConfig config;
  config.max_nodes = options.candidate_nodes;
  config.max_simulations = options.candidate_simulations;
  config.c_puct = options.c_puct;
  if (options.time_budget_ms != 0) {
    config.absolute_deadline =
        started + std::chrono::milliseconds(options.time_budget_ms);
  }
  candidate::NeuralPuctSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  const candidate::SearchStats stats = search.stats();

  Decision decision;
  decision.simulations = stats.simulations;
  decision.nodes = stats.nodes;
  decision.neural_evaluations = stats.neural_evaluations;
  decision.depth = stats.max_depth;
  decision.deadline_reached = stats.deadline_reached;
  decision.node_cap_reached = stats.node_cap_reached;
  if (decision.simulations > options.candidate_simulations) {
    throw std::logic_error("candidate exceeded its simulation cap");
  }
  if (decision.nodes > options.candidate_nodes) {
    throw std::logic_error("candidate exceeded its tree-node cap");
  }

  const ps::Player mover = state.to_move;
  for (const ps::Move move : moves) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("candidate returned an overlong action");
    }
    decision.action.push_back(candidate::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  verify_complete_action(before, state, decision.action);
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(candidate::SearchClock::now() -
                                                started)
          .count();
  return decision;
}

Decision choose_incumbent(ps::GameState &state, std::string_view transcript,
                          const Options &options) {
  const ps::GameState before = state;
  const auto started = incumbent::SearchClock::now();
  Decision decision;
  if (!incumbent::try_replay_correction(
          state, player_id(state.to_move), transcript, decision.action)) {
    incumbent::SearchConfig config;
    config.replay_value_blend_percent = 15;
    config.max_nodes = options.incumbent_nodes;
    if (options.time_budget_ms != 0) {
      config.absolute_deadline =
          started + std::chrono::milliseconds(options.time_budget_ms);
    }
    incumbent::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
    decision.nodes = search.stats().nodes;
    decision.depth = search.stats().completed_turn_depth;
    if (decision.nodes > options.incumbent_nodes) {
      throw std::logic_error("incumbent exceeded its search-node cap");
    }
    const ps::Player mover = state.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("incumbent returned an overlong action");
      }
      decision.action.push_back(
          incumbent::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
  }
  verify_complete_action(before, state, decision.action);
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(incumbent::SearchClock::now() -
                                                started)
          .count();
  return decision;
}

void add(Work &work, const Decision &decision) {
  ++work.decisions;
  work.searches += decision.nodes != 0 ? 1U : 0U;
  work.simulations += decision.simulations;
  work.nodes += decision.nodes;
  work.neural_evaluations += decision.neural_evaluations;
  work.depth_sum += decision.depth;
  work.deadlines += decision.deadline_reached ? 1U : 0U;
  work.node_caps += decision.node_cap_reached ? 1U : 0U;
  work.milliseconds += decision.milliseconds;
  work.maximum_milliseconds =
      std::max(work.maximum_milliseconds, decision.milliseconds);
}

void merge(Work &into, const Work &from) {
  into.decisions += from.decisions;
  into.searches += from.searches;
  into.simulations += from.simulations;
  into.nodes += from.nodes;
  into.neural_evaluations += from.neural_evaluations;
  into.depth_sum += from.depth_sum;
  into.deadlines += from.deadlines;
  into.node_caps += from.node_caps;
  into.milliseconds += from.milliseconds;
  into.maximum_milliseconds =
      std::max(into.maximum_milliseconds, from.maximum_milliseconds);
}

GameResult play(const Opening &opening, int candidate_player,
                const Options &options) {
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  GameResult result;
  result.opening_id = opening.id;
  result.candidate_player = candidate_player;
  while (!ps::is_terminal(state) && result.turns < options.maximum_turns) {
    const bool candidate_turn = player_id(state.to_move) == candidate_player;
    try {
      const Decision decision = candidate_turn
                                    ? choose_candidate(state, options)
                                    : choose_incumbent(state, transcript, options);
      append_turn(transcript, decision.action);
      add(candidate_turn ? result.candidate_work : result.incumbent_work,
          decision);
      ++result.turns;
    } catch (const std::exception &error) {
      result.illegal = true;
      result.illegal_engine = candidate_turn ? "candidate" : "incumbent";
      result.error = error.what();
      result.winner = candidate_turn ? 1 - candidate_player : candidate_player;
      return result;
    }
  }
  if (const std::optional<ps::Player> winning = ps::winner(state)) {
    result.winner = player_id(*winning);
  }
  return result;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      std::cout
          << "usage: comparison_gate [--bank FILE] [--output FILE|-] "
             "[--candidate-simulations N] [--candidate-nodes N] "
             "[--incumbent-nodes N] [--time-budget-ms N] "
             "[--max-turns N] [--opening-limit N] [--shard-count N] "
             "[--shard-index N] [--c-puct FLOAT]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " +
                                  std::string(argument));
    }
    const std::string value(argv[++index]);
    if (argument == "--bank") options.bank = value;
    else if (argument == "--output") options.output = value;
    else if (argument == "--candidate-simulations") {
      options.candidate_simulations =
          parse_integer<std::uint64_t>(value, "candidate simulations");
      options.candidate_simulations_explicit = true;
    } else if (argument == "--candidate-nodes") {
      options.candidate_nodes =
          parse_integer<std::size_t>(value, "candidate nodes");
    } else if (argument == "--incumbent-nodes") {
      options.incumbent_nodes =
          parse_integer<std::uint64_t>(value, "incumbent nodes");
      options.incumbent_nodes_explicit = true;
    } else if (argument == "--time-budget-ms") {
      options.time_budget_ms =
          parse_integer<std::uint32_t>(value, "time budget");
    } else if (argument == "--max-turns") {
      options.maximum_turns = parse_integer<int>(value, "maximum turns");
    } else if (argument == "--opening-limit") {
      options.opening_limit =
          parse_integer<std::size_t>(value, "opening limit");
    } else if (argument == "--shard-count") {
      options.shard_count = parse_integer<int>(value, "shard count");
    } else if (argument == "--shard-index") {
      options.shard_index =
          parse_integer<int>(value, "shard index", true);
    } else if (argument == "--c-puct") {
      options.c_puct = parse_float(value, "c-puct");
    } else {
      throw std::invalid_argument("unknown argument: " +
                                  std::string(argument));
    }
  }
  if (options.shard_index < 0 || options.shard_index >= options.shard_count) {
    throw std::invalid_argument("shard index is outside the shard count");
  }
  if (options.time_budget_ms != 0) {
    if (!options.candidate_simulations_explicit) {
      options.candidate_simulations = 1'000'000;
    }
    if (!options.incumbent_nodes_explicit) {
      options.incumbent_nodes = incumbent::kMaximumNodes;
    }
  }
  return options;
}

void write_work(std::ostream &out, const Work &work) {
  out << "{\"decisions\":" << work.decisions
      << ",\"searches\":" << work.searches
      << ",\"simulations\":" << work.simulations
      << ",\"nodes\":" << work.nodes
      << ",\"neural_evaluations\":" << work.neural_evaluations
      << ",\"depth_sum\":" << work.depth_sum
      << ",\"deadlines\":" << work.deadlines
      << ",\"node_caps\":" << work.node_caps
      << ",\"total_ms\":" << work.milliseconds
      << ",\"max_ms\":" << work.maximum_milliseconds << '}';
}

void write_game(std::ostream &out, const GameResult &game) {
  out << "{\"opening\":" << json_string(game.opening_id)
      << ",\"candidate_player\":" << game.candidate_player
      << ",\"winner\":";
  if (game.winner < 0) out << "null"; else out << game.winner;
  out << ",\"turns\":" << game.turns
      << ",\"illegal\":" << (game.illegal ? "true" : "false");
  if (game.illegal) {
    out << ",\"illegal_engine\":" << json_string(game.illegal_engine)
        << ",\"error\":" << json_string(game.error);
  }
  out << ",\"candidate\":";
  write_work(out, game.candidate_work);
  out << ",\"incumbent\":";
  write_work(out, game.incumbent_work);
  out << '}';
}

Summary summarize(const std::vector<GameResult> &games) {
  Summary summary;
  summary.games = static_cast<int>(games.size());
  for (const GameResult &game : games) {
    ++summary.color_games[game.candidate_player];
    if (game.winner < 0) {
      ++summary.unfinished;
    } else if (game.winner == game.candidate_player) {
      ++summary.candidate_wins;
      ++summary.color_wins[game.candidate_player];
    } else {
      ++summary.incumbent_wins;
    }
    if (game.illegal) {
      ++summary.illegal_actions;
      if (game.illegal_engine == "candidate") {
        ++summary.candidate_illegal_actions;
      } else {
        ++summary.incumbent_illegal_actions;
      }
    }
    merge(summary.candidate_work, game.candidate_work);
    merge(summary.incumbent_work, game.incumbent_work);
  }
  return summary;
}

void write_output(std::ostream &out, const Options &options,
                  const std::vector<GameResult> &games) {
  const Summary summary = summarize(games);
  out << std::setprecision(17)
      << "{\"schema\":\"papersoccer.neural-puct-comparison.v1\","
      << "\"configuration\":{\"opening_source\":"
      << json_string(options.bank.empty() ? "deterministic-defaults"
                                          : options.bank)
      << ",\"candidate_simulation_cap\":" << options.candidate_simulations
      << ",\"candidate_node_cap\":" << options.candidate_nodes
      << ",\"incumbent_node_cap\":" << options.incumbent_nodes
      << ",\"time_budget_ms\":" << options.time_budget_ms
      << ",\"construction_inclusive_deadline\":true"
      << ",\"maximum_turns\":" << options.maximum_turns
      << ",\"shard_count\":" << options.shard_count
      << ",\"shard_index\":" << options.shard_index
      << ",\"c_puct\":" << options.c_puct << "},"
      << "\"summary\":{\"games\":" << summary.games
      << ",\"candidate_wins\":" << summary.candidate_wins
      << ",\"incumbent_wins\":" << summary.incumbent_wins
      << ",\"unfinished\":" << summary.unfinished
      << ",\"illegal_actions\":" << summary.illegal_actions
      << ",\"candidate_illegal_actions\":"
      << summary.candidate_illegal_actions
      << ",\"incumbent_illegal_actions\":"
      << summary.incumbent_illegal_actions
      << ",\"colors\":{\"candidate_as_0\":{\"games\":"
      << summary.color_games[0] << ",\"wins\":" << summary.color_wins[0]
      << "},\"candidate_as_1\":{\"games\":" << summary.color_games[1]
      << ",\"wins\":" << summary.color_wins[1] << "}},\"candidate\":";
  write_work(out, summary.candidate_work);
  out << ",\"incumbent\":";
  write_work(out, summary.incumbent_work);
  out << "},\"games\":[";
  for (std::size_t index = 0; index < games.size(); ++index) {
    if (index != 0) out << ',';
    write_game(out, games[index]);
  }
  out << "]}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<Opening> openings =
        options.bank.empty() ? default_openings() : read_bank(options.bank);
    if (openings.size() > options.opening_limit) {
      openings.resize(options.opening_limit);
    }

    std::vector<GameResult> games;
    games.reserve(openings.size() * 2U);
    for (std::size_t index = 0; index < openings.size(); ++index) {
      if (static_cast<int>(index %
                           static_cast<std::size_t>(options.shard_count)) !=
          options.shard_index) {
        continue;
      }
      games.push_back(play(openings[index], 0, options));
      games.push_back(play(openings[index], 1, options));
    }
    if (games.empty()) {
      throw std::invalid_argument("selected shard has no openings");
    }

    if (options.output == "-") {
      write_output(std::cout, options, games);
    } else {
      std::ofstream output(options.output);
      if (!output) {
        throw std::runtime_error("could not open output file: " +
                                 options.output);
      }
      write_output(output, options, games);
      if (!output) {
        throw std::runtime_error("could not write output file: " +
                                 options.output);
      }
    }

    const Summary summary = summarize(games);
    return summary.unfinished == 0 && summary.illegal_actions == 0 ? 0 : 70;
  } catch (const std::invalid_argument &error) {
    std::cerr << "neural PUCT comparison gate: " << error.what() << '\n';
    return 64;
  } catch (const std::exception &error) {
    std::cerr << "neural PUCT comparison gate: " << error.what() << '\n';
    return 70;
  }
}
