#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank5_engine
#include "../rank_5/bot.cpp"
#undef turn_action_v2

#if defined(__GNUC__) && !defined(__clang__)
namespace papersoccer::shell_proof_engine {
namespace replay_book = ::papersoccer::rank5_engine::replay_book;
namespace replay_value_model =
    ::papersoccer::rank5_engine::replay_value_model;
}
#endif

#define turn_action_v2 shell_proof_engine
#include "bot.cpp"
#undef turn_action_v2

#include <array>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace incumbent = papersoccer::rank5_engine;
namespace candidate = papersoccer::shell_proof_engine;

namespace {

constexpr std::string_view kExpectedHeader =
    "opening_id\tsplit\tstratum\tsource_agent_id\tsource_game_id\t"
    "opponent_agent_id\twinner_player_id\tturn_index\tphysical_edges\t"
    "state_key\tcanonical_key\tball_x\tball_y\tmover\twinner_tier\t"
    "goal_distance_band\t"
    "used_edge_band\tshell_edge_band\topening_family\tobserved_winner_action\t"
    "transcript";

struct Options {
  std::string bank;
  std::string output;
  std::string stage;
  std::string manifest_sha256;
  std::string bank_sha256;
  std::string candidate_sha256;
  std::string incumbent_sha256;
  std::string runner_sha256;
  std::uint64_t node_budget{5'000};
  std::uint32_t time_budget_ms{};
  int maximum_turns{320};
  int shard_count{1};
  int shard_index{};
};

struct Opening {
  std::string id;
  std::string split;
  std::string stratum;
  std::string winner_tier;
  std::string observed_action;
  std::string transcript;
  std::string canonical_key;
  std::uint64_t source_game_id{};
  int historical_winner{};
  int expected_edges{};
  int expected_ball_x{};
  int expected_ball_y{};
  int expected_mover{};
  ps::GameState state;
};

struct Decision {
  std::string action;
  std::uint64_t nodes{};
  std::uint32_t completed_depth{};
  int root_score{};
  double milliseconds{};
  std::uint64_t rebound_goal_probes{};
  std::uint64_t rebound_goal_hits{};
  std::uint64_t rebound_loss_hits{};
};

struct GameResult {
  int candidate_player{};
  int winner{-1};
  int turns{};
  std::uint64_t candidate_nodes{};
  std::uint64_t incumbent_nodes{};
  double candidate_ms{};
  double incumbent_ms{};
  std::uint64_t candidate_searches{};
  std::uint64_t incumbent_searches{};
  std::uint64_t candidate_depth_sum{};
  std::uint64_t incumbent_depth_sum{};
  std::uint64_t rebound_goal_probes{};
  std::uint64_t rebound_goal_hits{};
  std::uint64_t rebound_loss_hits{};
};

struct ControlResult {
  int winner{-1};
  int turns{};
};

struct PairResult {
  Opening opening;
  std::array<GameResult, 2> games{};
  ControlResult incumbent_control;
  double candidate_score{};
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
                      bool allow_zero = true) {
  Integer value{};
  const char *begin = raw.data();
  const char *end = begin + raw.size();
  const auto [position, error] = std::from_chars(begin, end, value);
  if (raw.empty() || error != std::errc{} || position != end ||
      (!allow_zero && value == 0)) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return value;
}

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) transcript.push_back('/');
  transcript.append(action);
}

void verify_complete_action(const ps::GameState &before,
                            const ps::GameState &after,
                            std::string_view action) {
  if (action.empty()) {
    throw std::logic_error("engine returned an empty action");
  }
  ps::GameState replayed = before;
  incumbent::apply_encoded_turn(replayed, action);
  if (replayed.ball != after.ball || replayed.to_move != after.to_move ||
      replayed.status != after.status ||
      replayed.used_segments != after.used_segments ||
      replayed.visit_count != after.visit_count) {
    throw std::logic_error("engine action did not reproduce its state");
  }
}

Decision choose_incumbent(ps::GameState &state, std::string_view transcript,
                          std::uint64_t node_budget,
                          std::uint32_t time_budget_ms) {
  const ps::GameState before = state;
  Decision decision;
  if (!incumbent::try_replay_correction(
          state, player_id(state.to_move), transcript, decision.action)) {
    incumbent::SearchConfig config;
    config.replay_value_blend_percent = 15;
    if (time_budget_ms != 0) config.max_time_ms = time_budget_ms;
    else config.max_nodes = node_budget;
    const auto started = incumbent::SearchClock::now();
    incumbent::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> action = search.run();
    decision.milliseconds =
        std::chrono::duration<double, std::milli>(
            incumbent::SearchClock::now() - started).count();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : action) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("incumbent returned an overlong action");
      }
      decision.action.push_back(incumbent::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    decision.nodes = search.stats().nodes;
    decision.completed_depth = search.stats().completed_turn_depth;
    decision.root_score = search.stats().root_score;
  }
  verify_complete_action(before, state, decision.action);
  return decision;
}

Decision choose_candidate(ps::GameState &state, std::string_view transcript,
                          std::uint64_t node_budget,
                          std::uint32_t time_budget_ms) {
  const ps::GameState before = state;
  Decision decision;
  if (!candidate::try_replay_correction(
          state, player_id(state.to_move), transcript, decision.action)) {
    candidate::SearchConfig config;
    config.replay_value_blend_percent = 15;
    if (time_budget_ms != 0) config.max_time_ms = time_budget_ms;
    else config.max_nodes = node_budget;
    const auto started = candidate::SearchClock::now();
    candidate::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> action = search.run();
    decision.milliseconds =
        std::chrono::duration<double, std::milli>(
            candidate::SearchClock::now() - started).count();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : action) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("candidate returned an overlong action");
      }
      decision.action.push_back(candidate::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    decision.nodes = search.stats().nodes;
    decision.completed_depth = search.stats().completed_turn_depth;
    decision.root_score = search.stats().root_score;
    decision.rebound_goal_probes = search.stats().rebound_goal_probes;
    decision.rebound_goal_hits = search.stats().rebound_goal_hits;
    decision.rebound_loss_hits = search.stats().rebound_loss_hits;
  }
  verify_complete_action(before, state, decision.action);
  return decision;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size() : separator;
    incumbent::apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) break;
    begin = separator + 1U;
  }
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("bank contains a terminal opening");
  }
  return state;
}

std::vector<Opening> read_bank(const std::string &path) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("could not open bank: " + path);
  std::string line;
  if (!std::getline(input, line) || line != kExpectedHeader) {
    throw std::invalid_argument("bank has an unexpected header");
  }
  std::vector<Opening> openings;
  std::size_t line_number = 1;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) {
      throw std::invalid_argument("bank contains an empty row");
    }
    const std::vector<std::string_view> fields = split_tabs(line);
    if (fields.size() != 21) {
      throw std::invalid_argument("bank row has the wrong field count");
    }
    Opening opening;
    opening.id = fields[0];
    opening.split = fields[1];
    opening.stratum = fields[2];
    opening.source_game_id = parse_integer<std::uint64_t>(fields[4], "game id");
    opening.historical_winner = parse_integer<int>(fields[6], "historical winner");
    opening.expected_edges = parse_integer<int>(fields[8], "physical edges");
    opening.canonical_key = fields[10];
    opening.expected_ball_x = parse_integer<int>(fields[11], "ball x");
    opening.expected_ball_y = parse_integer<int>(fields[12], "ball y");
    opening.expected_mover = parse_integer<int>(fields[13], "mover");
    opening.winner_tier = fields[14];
    opening.observed_action = fields[19];
    opening.transcript = fields[20] == "-" ? std::string{} : std::string(fields[20]);
    opening.state = reconstruct(opening.transcript);
    if (opening.state.ball.x != opening.expected_ball_x ||
        opening.state.ball.y != opening.expected_ball_y ||
        player_id(opening.state.to_move) != opening.expected_mover ||
        opening.historical_winner < 0 || opening.historical_winner > 1 ||
        static_cast<int>(opening.state.used_segments.size()) !=
            opening.expected_edges) {
      throw std::invalid_argument("bank metadata does not match replayed state at line " +
                                  std::to_string(line_number));
    }
    openings.push_back(std::move(opening));
  }
  if (openings.empty()) throw std::invalid_argument("bank is empty");
  return openings;
}

GameResult play(const Opening &opening, int candidate_player,
                std::uint64_t node_budget, std::uint32_t time_budget_ms,
                int maximum_turns) {
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  GameResult result;
  result.candidate_player = candidate_player;
  while (!ps::is_terminal(state) && result.turns < maximum_turns) {
    const bool candidate_turn = player_id(state.to_move) == candidate_player;
    const Decision decision = candidate_turn
        ? choose_candidate(state, transcript, node_budget, time_budget_ms)
        : choose_incumbent(state, transcript, node_budget, time_budget_ms);
    append_turn(transcript, decision.action);
    if (candidate_turn) {
      result.candidate_nodes += decision.nodes;
      result.candidate_ms += decision.milliseconds;
      result.candidate_searches += decision.nodes != 0 ? 1U : 0U;
      result.candidate_depth_sum += decision.completed_depth;
      result.rebound_goal_probes += decision.rebound_goal_probes;
      result.rebound_goal_hits += decision.rebound_goal_hits;
      result.rebound_loss_hits += decision.rebound_loss_hits;
    } else {
      result.incumbent_nodes += decision.nodes;
      result.incumbent_ms += decision.milliseconds;
      result.incumbent_searches += decision.nodes != 0 ? 1U : 0U;
      result.incumbent_depth_sum += decision.completed_depth;
    }
    ++result.turns;
  }
  if (const std::optional<ps::Player> winning = ps::winner(state)) {
    result.winner = player_id(*winning);
  }
  return result;
}

ControlResult play_incumbent_control(const Opening &opening,
                                     std::uint64_t node_budget,
                                     std::uint32_t time_budget_ms,
                                     int maximum_turns) {
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  ControlResult result;
  while (!ps::is_terminal(state) && result.turns < maximum_turns) {
    const Decision decision = choose_incumbent(
        state, transcript, node_budget, time_budget_ms
    );
    append_turn(transcript, decision.action);
    ++result.turns;
  }
  if (const std::optional<ps::Player> winning = ps::winner(state)) {
    result.winner = player_id(*winning);
  }
  return result;
}

PairResult play_pair(const Opening &opening, const Options &options) {
  PairResult result;
  result.opening = opening;
  result.games[0] = play(opening, 0, options.node_budget,
                         options.time_budget_ms, options.maximum_turns);
  result.games[1] = play(opening, 1, options.node_budget,
                         options.time_budget_ms, options.maximum_turns);
  result.incumbent_control = play_incumbent_control(
      opening, options.node_budget, options.time_budget_ms,
      options.maximum_turns
  );
  if (result.games[0].winner >= 0 && result.games[1].winner >= 0) {
    result.candidate_score =
        ((result.games[0].winner == 0 ? 1.0 : 0.0) +
         (result.games[1].winner == 1 ? 1.0 : 0.0)) / 2.0;
  } else {
    result.candidate_score = -1.0;
  }
  return result;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--help") {
      std::cout << "usage: comparison_gate --bank FILE --stage NAME "
                   "--node-budget N --max-turns N --shard-count N "
                   "--shard-index N --output FILE [--time-budget-ms N] "
                   "[identity hashes]\n";
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value after " + std::string(argument));
    }
    const std::string value(argv[++index]);
    if (argument == "--bank") options.bank = value;
    else if (argument == "--output") options.output = value;
    else if (argument == "--stage") options.stage = value;
    else if (argument == "--node-budget") {
      options.node_budget = parse_integer<std::uint64_t>(value, "node budget", false);
    } else if (argument == "--max-turns") {
      options.maximum_turns = parse_integer<int>(value, "maximum turns", false);
    } else if (argument == "--time-budget-ms") {
      options.time_budget_ms =
          parse_integer<std::uint32_t>(value, "time budget", false);
    } else if (argument == "--shard-count") {
      options.shard_count = parse_integer<int>(value, "shard count", false);
    } else if (argument == "--shard-index") {
      options.shard_index = parse_integer<int>(value, "shard index");
    } else if (argument == "--manifest-sha256") options.manifest_sha256 = value;
    else if (argument == "--bank-sha256") options.bank_sha256 = value;
    else if (argument == "--candidate-sha256") options.candidate_sha256 = value;
    else if (argument == "--incumbent-sha256") options.incumbent_sha256 = value;
    else if (argument == "--runner-sha256") options.runner_sha256 = value;
    else throw std::invalid_argument("unknown argument: " + std::string(argument));
  }
  if (options.bank.empty() || options.stage.empty() || options.output.empty() ||
      options.shard_index < 0 || options.shard_index >= options.shard_count) {
    throw std::invalid_argument("incomplete or inconsistent comparison options");
  }
  return options;
}

void write_game(std::ostream &out, const GameResult &game) {
  out << "{\"candidate_player\":" << game.candidate_player
      << ",\"winner\":";
  if (game.winner < 0) out << "null"; else out << game.winner;
  out << ",\"turns\":" << game.turns
      << ",\"candidate_nodes\":" << game.candidate_nodes
      << ",\"incumbent_nodes\":" << game.incumbent_nodes
      << ",\"candidate_ms\":" << game.candidate_ms
      << ",\"incumbent_ms\":" << game.incumbent_ms
      << ",\"candidate_searches\":" << game.candidate_searches
      << ",\"incumbent_searches\":" << game.incumbent_searches
      << ",\"candidate_depth_sum\":" << game.candidate_depth_sum
      << ",\"incumbent_depth_sum\":" << game.incumbent_depth_sum
      << ",\"rebound_goal_probes\":" << game.rebound_goal_probes
      << ",\"rebound_goal_hits\":" << game.rebound_goal_hits
      << ",\"rebound_loss_hits\":" << game.rebound_loss_hits << '}';
}

void write_output(const Options &options, const std::vector<PairResult> &pairs,
                  int unfinished) {
  std::ofstream out(options.output);
  if (!out) throw std::runtime_error("could not open output file");
  out << std::setprecision(17)
      << "{\n  \"schema\":\"papersoccer.codingame-promotion-shard.v2\",\n"
      << "  \"identity\":{\"manifest_sha256\":" << json_string(options.manifest_sha256)
      << ",\"bank_sha256\":" << json_string(options.bank_sha256)
      << ",\"candidate_submission_sha256\":" << json_string(options.candidate_sha256)
      << ",\"incumbent_submission_sha256\":" << json_string(options.incumbent_sha256)
      << ",\"runner_sha256\":" << json_string(options.runner_sha256) << "},\n"
      << "  \"configuration\":{\"stage\":" << json_string(options.stage)
      << ",\"rules\":{\"width\":8,\"height\":10,"
         "\"goal_rule\":\"own_goals_allowed\",\"blocked_rule\":\"mover_loses\"},"
      << "\"node_budget\":" << options.node_budget
      << ",\"time_budget_ms\":" << options.time_budget_ms
      << ",\"maximum_turns\":" << options.maximum_turns
      << ",\"shard_count\":" << options.shard_count
      << ",\"shard_index\":" << options.shard_index << "},\n"
      << "  \"pairs\":[";
  for (std::size_t index = 0; index < pairs.size(); ++index) {
    if (index != 0) out << ',';
    const PairResult &pair = pairs[index];
    out << "{\"opening_id\":" << json_string(pair.opening.id)
        << ",\"stratum\":" << json_string(pair.opening.stratum)
        << ",\"winner_tier\":" << json_string(pair.opening.winner_tier)
        << ",\"canonical_key\":" << json_string(pair.opening.canonical_key)
        << ",\"source_game_id\":" << pair.opening.source_game_id
        << ",\"historical_winner_player\":"
        << pair.opening.historical_winner
        << ",\"observed_winner_action\":"
        << json_string(pair.opening.observed_action) << ",\"games\":[";
    write_game(out, pair.games[0]);
    out << ',';
    write_game(out, pair.games[1]);
    out << "],\"incumbent_control\":{\"winner\":";
    if (pair.incumbent_control.winner < 0) {
      out << "null";
    } else {
      out << pair.incumbent_control.winner;
    }
    out << ",\"turns\":" << pair.incumbent_control.turns
        << "},\"candidate_pair_score\":";
    if (pair.candidate_score < 0.0) out << "null";
    else out << pair.candidate_score;
    out << '}';
  }
  out << "],\n  \"operational\":{\"illegal_actions\":0,\"empty_actions\":0,"
         "\"incomplete_actions\":0,\"overlong_actions\":0,"
      << "\"unfinished_games\":" << unfinished << "}\n}\n";
  if (!out) throw std::runtime_error("could not write output file");
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::vector<Opening> bank = read_bank(options.bank);
    std::vector<PairResult> pairs;
    int unfinished = 0;
    for (std::size_t index = 0; index < bank.size(); ++index) {
      if (static_cast<int>(index % static_cast<std::size_t>(options.shard_count)) !=
          options.shard_index) {
        continue;
      }
      PairResult pair = play_pair(bank[index], options);
      unfinished += pair.games[0].winner < 0 ? 1 : 0;
      unfinished += pair.games[1].winner < 0 ? 1 : 0;
      unfinished += pair.incumbent_control.winner < 0 ? 1 : 0;
      pairs.push_back(std::move(pair));
    }
    write_output(options, pairs, unfinished);
    return unfinished == 0 ? 0 : 70;
  } catch (const std::invalid_argument &error) {
    std::cerr << "comparison gate: " << error.what() << '\n';
    return 64;
  } catch (const std::exception &error) {
    std::cerr << "comparison gate: " << error.what() << '\n';
    return 70;
  }
}
