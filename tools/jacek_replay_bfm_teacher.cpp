#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 jacek_replay_teacher_engine
#include "../submissions/codingame/bots/rank_4/bot.cpp"
#undef turn_action_v2

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace teacher = papersoccer::jacek_replay_teacher_engine;

namespace {

struct Options {
  std::uint64_t nodes{32'000};
  std::uint64_t deep_nodes{};
  std::size_t deep_percent{10};
  std::size_t max_samples{100};
};

struct InputRecord {
  std::string group_id;
  std::string source;
  int winner{-1};
  std::vector<std::string> actions;
};

struct Label {
  ps::GameState state;
  std::vector<int> prefix_players;
  std::vector<std::string> prefix_actions;
  int root_score{};
  std::uint32_t completed_depth{};
  std::uint64_t nodes{};
  std::optional<int> proven_winner{};
};

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
          constexpr char hex[] = "0123456789abcdef";
          out << "\\u00" << hex[character >> 4U] << hex[character & 15U];
        } else {
          out << character;
        }
    }
  }
  out << '"';
  return out.str();
}

template <typename UInt>
UInt parse_unsigned(std::string_view raw, std::string_view label) {
  UInt result{};
  const auto [end, error] =
      std::from_chars(raw.data(), raw.data() + raw.size(), result);
  if (raw.empty() || error != std::errc{} || end != raw.data() + raw.size() ||
      result == 0) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return result;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help") {
      std::cout << "usage: papersoccer_jacek_replay_teacher "
                   "[--nodes N] [--deep-nodes N] [--deep-percent N] "
                   "[--max-samples N]\n"
                   "stdin TSV: group_id<TAB>source<TAB>winner<TAB>transcript\n";
      std::exit(0);
    }
    if (++index >= argc) {
      throw std::invalid_argument("missing value for " + std::string(option));
    }
    const std::string_view value(argv[index]);
    if (option == "--nodes") {
      options.nodes = parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--deep-nodes") {
      options.deep_nodes = parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--deep-percent") {
      options.deep_percent = parse_unsigned<std::size_t>(value, option);
      if (options.deep_percent > 100U) {
        throw std::invalid_argument("--deep-percent must not exceed 100");
      }
    } else if (option == "--max-samples") {
      options.max_samples = parse_unsigned<std::size_t>(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  return options;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = value.find(separator, begin);
    result.push_back(value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return result;
    begin = end + 1;
  }
}

InputRecord parse_record(std::string_view line) {
  const std::vector<std::string_view> fields = split(line, '\t');
  if (fields.size() != 4 || fields[0].empty() || fields[1].empty()) {
    throw std::invalid_argument("teacher input must contain four TSV fields");
  }
  InputRecord record;
  record.group_id = fields[0];
  record.source = fields[1];
  if (fields[2] == "0") record.winner = 0;
  else if (fields[2] == "1") record.winner = 1;
  else throw std::invalid_argument("winner must be zero or one");
  if (!fields[3].empty()) {
    for (const std::string_view action : split(fields[3], '/')) {
      if (action.empty() ||
          std::any_of(action.begin(), action.end(),
                      [](char value) { return value < '0' || value > '7'; })) {
        throw std::invalid_argument("transcript contains an invalid action");
      }
      record.actions.emplace_back(action);
    }
  }
  if (record.actions.empty()) {
    throw std::invalid_argument("transcript must contain at least one turn");
  }
  return record;
}

ps::RulesConfig contest_rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

std::set<std::size_t> sampled_boundaries(std::size_t count,
                                         std::size_t maximum) {
  std::set<std::size_t> result;
  const std::size_t retained = std::min(count, maximum);
  for (std::size_t index = 0; index < retained; ++index) {
    result.insert(index * count / retained);
  }
  return result;
}

void write_prefix(std::ostream &out, const std::vector<int> &players,
                  const std::vector<std::string> &actions) {
  out << '[';
  for (std::size_t index = 0; index < actions.size(); ++index) {
    if (index != 0) out << ',';
    out << "{\"player_id\":" << players[index]
        << ",\"action\":" << json_string(actions[index]) << '}';
  }
  out << ']';
}

bool search_label(const ps::GameState &state, std::uint64_t nodes,
                  Label &label) {
  teacher::SearchConfig config;
  config.max_nodes = nodes;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  teacher::CompleteTurnSearch search(state, config);
  (void)search.run();
  const teacher::SearchStats &stats = search.stats();
  if (stats.completed_turn_depth == 0 || stats.nodes == 0 ||
      !std::isfinite(static_cast<double>(stats.root_score))) {
    return false;
  }
  label.root_score = stats.root_score;
  label.completed_depth = stats.completed_turn_depth;
  label.nodes = stats.nodes;
  label.proven_winner.reset();
  if (stats.root_score >=
      teacher::kMateScore - static_cast<int>(teacher::kMaximumTurnDepth)) {
    label.proven_winner = 0;
  } else if (stats.root_score <=
             -teacher::kMateScore +
                 static_cast<int>(teacher::kMaximumTurnDepth)) {
    label.proven_winner = 1;
  }
  return true;
}

void write_label(const InputRecord &record, const Label &label) {
  std::cout << "{\"schema\":\"papersoccer.jacek-replay-teacher.v1\","
            << "\"group_id\":" << json_string(record.group_id) << ','
            << "\"source\":" << json_string(record.source) << ','
            << "\"winner\":" << record.winner << ",\"prefix\":";
  write_prefix(std::cout, label.prefix_players, label.prefix_actions);
  std::cout << ",\"mover\":" << player_id(label.state.to_move)
            << ",\"root_score\":" << label.root_score
            << ",\"completed_depth\":" << label.completed_depth
            << ",\"nodes\":" << label.nodes
            << ",\"proven_winner\":";
  if (label.proven_winner.has_value()) std::cout << *label.proven_winner;
  else std::cout << "null";
  std::cout << ",\"weight\":1.0}\n";
}

void relabel(const InputRecord &record, const Options &options) {
  // Validate the complete record before emitting any rows so a malformed game
  // can never leave a partially accepted teacher shard behind.
  ps::GameState verified = ps::make_initial_state(contest_rules());
  for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
    if (ps::is_terminal(verified)) {
      throw std::invalid_argument("transcript continues after terminal state");
    }
    try {
      teacher::apply_encoded_turn(verified, record.actions[turn]);
    } catch (const std::exception &error) {
      throw std::invalid_argument("turn " + std::to_string(turn) +
                                  " action " + record.actions[turn] +
                                  ": " + error.what());
    }
  }
  if (!ps::is_terminal(verified)) {
    throw std::invalid_argument("transcript does not end in a terminal state");
  }
  const std::optional<ps::Player> verified_winner = ps::winner(verified);
  if (!verified_winner.has_value() ||
      player_id(*verified_winner) != record.winner) {
    throw std::invalid_argument("declared winner does not match transcript");
  }

  ps::GameState state = ps::make_initial_state(contest_rules());
  std::vector<int> prefix_players;
  std::vector<std::string> prefix_actions;
  std::vector<Label> labels;
  const std::set<std::size_t> selected =
      sampled_boundaries(record.actions.size(), options.max_samples);

  for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
    if (ps::is_terminal(state)) {
      throw std::invalid_argument("transcript continues after terminal state");
    }
    if (selected.contains(turn)) {
      Label label{state, prefix_players, prefix_actions};
      if (search_label(state, options.nodes, label)) {
        labels.push_back(std::move(label));
      }
    }

    const int mover = player_id(state.to_move);
    teacher::apply_encoded_turn(state, record.actions[turn]);
    prefix_players.push_back(mover);
    prefix_actions.push_back(record.actions[turn]);
  }
  if (state.status != verified.status || state.ball != verified.ball ||
      state.to_move != verified.to_move ||
      state.used_segments != verified.used_segments) {
    throw std::logic_error("relabel replay differs from validation replay");
  }

  if (options.deep_nodes != 0 && !labels.empty()) {
    std::vector<std::size_t> order(labels.size());
    for (std::size_t index = 0; index < order.size(); ++index) {
      order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&](std::size_t left,
                                                      std::size_t right) {
      const int left_uncertainty = std::abs(labels[left].root_score);
      const int right_uncertainty = std::abs(labels[right].root_score);
      if (left_uncertainty != right_uncertainty) {
        return left_uncertainty < right_uncertainty;
      }
      return labels[left].prefix_actions.size() <
             labels[right].prefix_actions.size();
    });
    const std::size_t deep_count = std::max<std::size_t>(
        1U, (labels.size() * options.deep_percent + 99U) / 100U);
    for (std::size_t selected_index = 0;
         selected_index < std::min(deep_count, order.size());
         ++selected_index) {
      Label deep = labels[order[selected_index]];
      if (search_label(deep.state, options.deep_nodes, deep)) {
        labels[order[selected_index]] = std::move(deep);
      }
    }
  }
  for (const Label &label : labels) {
    write_label(record, label);
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(std::cin, line)) {
      ++line_number;
      if (line.empty() || line.front() == '#' ||
          line == "group_id\tsource\twinner\ttranscript") {
        continue;
      }
      try {
        relabel(parse_record(line), options);
      } catch (const std::exception &error) {
        throw std::runtime_error("line " + std::to_string(line_number) +
                                 ": " + error.what());
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek replay teacher: " << error.what() << '\n';
    return 2;
  }
}
