#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 jacek_replay_rank4_position_teacher_engine
#include "../submissions/codingame/bots/rank_4/bot.cpp"
#undef turn_action_v2

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef PAPERSOCCER_JACEK_REPLAY_RANK4_POSITION_TEACHER_SOURCE_SHA256
#error "Rank-4 position-teacher source closure SHA-256 must be provided by CMake"
#endif

namespace ps = papersoccer;
namespace teacher = papersoccer::jacek_replay_rank4_position_teacher_engine;

namespace {

constexpr std::string_view kSchema = "papersoccer.jacek-replay-teacher.v1";
constexpr std::string_view kHeader =
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix";

struct Options {
  std::string campaign_id;
  std::uint64_t nodes{32'000U};
  std::uint32_t time_ms{60'000U};
};

struct PositionRow {
  std::string position_id;
  std::string root_group_id;
  std::string group_id;
  std::string source;
  std::string split;
  int winner{};
  int mover{};
  std::vector<int> prefix_players;
  std::vector<std::string> actions;
  ps::GameState state;
};

bool is_lower_sha256(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void require_text(std::string_view value, std::string_view label) {
  if (value.empty() ||
      std::any_of(value.begin(), value.end(), [](unsigned char character) {
        return character < 0x20U || character == 0x7fU;
      })) {
    throw std::invalid_argument(std::string(label) +
                                " must be nonempty printable text");
  }
}

template <typename UInt>
UInt parse_unsigned(std::string_view raw, std::string_view label) {
  UInt value{};
  const auto [end, error] =
      std::from_chars(raw.data(), raw.data() + raw.size(), value);
  if (raw.empty() || error != std::errc{} || end != raw.data() + raw.size() ||
      value == 0U) {
    throw std::invalid_argument(std::string(label) +
                                " requires a positive integer");
  }
  return value;
}

std::string require_value(int &index, int argc, char **argv,
                          std::string_view option) {
  if (++index >= argc) {
    throw std::invalid_argument("missing value for " + std::string(option));
  }
  return argv[index];
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help") {
      std::cout
          << "usage: papersoccer_jacek_replay_rank4_position_teacher "
             "--campaign-id ID [--nodes N] [--time-ms N]\n"
             "stdin TSV header: "
          << kHeader << '\n';
      std::exit(0);
    }
    const std::string value = require_value(index, argc, argv, option);
    if (option == "--campaign-id") {
      options.campaign_id = value;
    } else if (option == "--nodes") {
      options.nodes = parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--time-ms") {
      options.time_ms = parse_unsigned<std::uint32_t>(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  require_text(options.campaign_id, "--campaign-id");
  if (options.nodes > teacher::kMaximumNodes || options.time_ms > 60'000U) {
    throw std::invalid_argument("invalid Rank-4 teacher configuration");
  }
  return options;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0U;
  while (true) {
    const std::size_t end = value.find(separator, begin);
    result.push_back(value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return result;
    begin = end + 1U;
  }
}

ps::RulesConfig contest_rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

int parse_player(std::string_view raw, std::string_view label) {
  if (raw == "0") return 0;
  if (raw == "1") return 1;
  throw std::invalid_argument(std::string(label) + " must be zero or one");
}

PositionRow parse_row(std::string_view line) {
  const std::vector<std::string_view> fields = split(line, '\t');
  if (fields.size() != 8U) {
    throw std::invalid_argument(
        "position input must contain exactly eight TSV fields");
  }
  require_text(fields[0], "position_id");
  require_text(fields[1], "root_group_id");
  require_text(fields[2], "group_id");
  require_text(fields[3], "source");
  if (fields[4] != "train" && fields[4] != "validation" &&
      fields[4] != "test") {
    throw std::invalid_argument("split must be train, validation, or test");
  }

  PositionRow row;
  row.position_id = fields[0];
  row.root_group_id = fields[1];
  row.group_id = fields[2];
  row.source = fields[3];
  row.split = fields[4];
  row.winner = parse_player(fields[5], "winner");
  row.mover = parse_player(fields[6], "mover");
  row.state = ps::make_initial_state(contest_rules());
  if (!fields[7].empty()) {
    const std::vector<std::string_view> actions = split(fields[7], '/');
    for (std::size_t turn = 0; turn < actions.size(); ++turn) {
      if (actions[turn].empty()) {
        throw std::invalid_argument("prefix contains an empty turn");
      }
      row.prefix_players.push_back(player_id(row.state.to_move));
      try {
        teacher::apply_encoded_turn(row.state, actions[turn]);
      } catch (const std::exception &error) {
        throw std::invalid_argument("prefix turn " + std::to_string(turn) +
                                    ": " + error.what());
      }
      row.actions.emplace_back(actions[turn]);
    }
  }
  if (ps::is_terminal(row.state)) {
    throw std::invalid_argument("prefix resolves to a terminal position");
  }
  if (player_id(row.state.to_move) != row.mover) {
    throw std::invalid_argument(
        "declared mover does not match the replayed prefix");
  }
  return row;
}

std::vector<PositionRow> read_rows(std::istream &input) {
  std::string line;
  if (!std::getline(input, line) || line != kHeader) {
    throw std::invalid_argument(
        "position input must begin with the exact TSV header");
  }
  std::vector<PositionRow> rows;
  std::set<std::string> position_ids;
  std::size_t line_number = 1U;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      throw std::invalid_argument("line " + std::to_string(line_number) +
                                  ": empty rows are not allowed");
    }
    try {
      PositionRow row = parse_row(line);
      if (!position_ids.insert(row.position_id).second) {
        throw std::invalid_argument("duplicate position_id");
      }
      rows.push_back(std::move(row));
    } catch (const std::exception &error) {
      throw std::invalid_argument("line " + std::to_string(line_number) +
                                  ": " + error.what());
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("could not read complete position input");
  }
  if (rows.empty()) {
    throw std::invalid_argument("position input contains no data rows");
  }
  return rows;
}

std::string json_string(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          constexpr char hex[] = "0123456789abcdef";
          output << "\\u00" << hex[character >> 4U]
                 << hex[character & 0x0fU];
        } else {
          output << character;
        }
    }
  }
  output << '"';
  return output.str();
}

void write_prefix(std::ostream &output, const PositionRow &row) {
  output << '[';
  for (std::size_t index = 0; index < row.actions.size(); ++index) {
    if (index != 0U) output << ',';
    output << "{\"player_id\":" << row.prefix_players[index]
           << ",\"action\":" << json_string(row.actions[index]) << '}';
  }
  output << ']';
}

void write_bool(std::ostream &output, bool value) {
  output << (value ? "true" : "false");
}

void label(const PositionRow &row, const Options &options,
           std::ostream &output) {
  teacher::SearchConfig config;
  config.max_nodes = options.nodes;
  config.max_time_ms = options.time_ms;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  teacher::CompleteTurnSearch search(row.state, config);
  (void)search.run();
  const teacher::SearchStats &stats = search.stats();
  const bool root_solved =
      std::abs(stats.root_score) >=
      teacher::kMateScore - static_cast<int>(teacher::kMaximumTurnDepth);
  const bool deadline_reached =
      stats.budget_exhausted && stats.nodes < options.nodes;
  const bool node_cap_reached =
      stats.budget_exhausted && stats.nodes == options.nodes;
  if (deadline_reached) {
    throw std::runtime_error("Rank-4 teacher reached its deadline");
  }
  if (stats.completed_turn_depth == 0U || stats.nodes == 0U) {
    throw std::runtime_error("Rank-4 teacher did not complete depth one");
  }
  if (!root_solved && !node_cap_reached &&
      stats.completed_turn_depth != teacher::kMaximumTurnDepth) {
    throw std::runtime_error(
        "Rank-4 teacher stopped before fixed work completed");
  }
  const std::optional<int> proven_winner =
      root_solved ? std::optional<int>(stats.root_score > 0 ? 0 : 1)
                  : std::nullopt;

  output << "{\"schema\":" << json_string(kSchema)
         << ",\"campaign_id\":" << json_string(options.campaign_id)
         << ",\"position_id\":" << json_string(row.position_id)
         << ",\"root_group_id\":" << json_string(row.root_group_id)
         << ",\"group_id\":" << json_string(row.group_id)
         << ",\"source\":" << json_string(row.source)
         << ",\"split\":" << json_string(row.split)
         << ",\"winner\":" << row.winner << ",\"prefix\":";
  write_prefix(output, row);
  output << ",\"mover\":" << row.mover
         << ",\"teacher\":{\"kind\":\"rank4-fixed-work\","
            "\"source_sha256\":"
         << json_string(
                PAPERSOCCER_JACEK_REPLAY_RANK4_POSITION_TEACHER_SOURCE_SHA256)
         << "},\"search_config\":{\"max_nodes\":" << options.nodes
         << ",\"max_time_ms\":" << options.time_ms
         << ",\"max_turn_depth\":" << teacher::kMaximumTurnDepth
         << ",\"replay_value_blend_percent\":15,"
            "\"teacher_residual_weight_percent\":100}"
         << ",\"search_stats\":{\"attempted_depth\":"
         << stats.attempted_turn_depth << ",\"completed_depth\":"
         << stats.completed_turn_depth << ",\"nodes\":" << stats.nodes
         << ",\"leaf_evaluations\":" << stats.leaf_evaluations
         << ",\"terminal_nodes\":" << stats.terminal_nodes
         << ",\"completed_actions\":" << stats.completed_actions
         << ",\"budget_exhausted\":";
  write_bool(output, stats.budget_exhausted);
  output << ",\"node_cap_reached\":";
  write_bool(output, node_cap_reached);
  output << ",\"deadline_reached\":";
  write_bool(output, deadline_reached);
  output << "},\"root_score\":" << stats.root_score
         << ",\"completed_depth\":" << stats.completed_turn_depth
         << ",\"nodes\":" << stats.nodes << ",\"root_solved\":";
  write_bool(output, root_solved);
  output << ",\"proven_winner\":";
  if (proven_winner.has_value()) output << *proven_winner;
  else output << "null";
  output << ",\"weight\":1.0}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (!is_lower_sha256(
            PAPERSOCCER_JACEK_REPLAY_RANK4_POSITION_TEACHER_SOURCE_SHA256)) {
      throw std::logic_error(
          "Rank-4 position-teacher source closure identity is invalid");
    }
    const Options options = parse_options(argc, argv);
    const std::vector<PositionRow> rows = read_rows(std::cin);
    for (const PositionRow &row : rows) label(row, options, std::cout);
    std::cout.flush();
    if (!std::cout) {
      throw std::runtime_error(
          "could not write complete Rank-4 teacher output");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek Rank-4 position teacher: " << error.what() << '\n';
    return 2;
  }
}
