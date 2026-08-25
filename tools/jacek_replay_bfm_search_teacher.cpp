#include "jacek_replay_bfm_search_teacher_internal.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"
#include "papersoccer/geometry.hpp"
#include "papersoccer/rules.hpp"

#ifndef PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256
#error "search teacher source closure SHA-256 must be provided by CMake"
#endif

namespace ps = papersoccer;

namespace papersoccer::jacek_replay_search_teacher {
namespace {

constexpr std::string_view kSchema =
    "papersoccer.jacek-replay-search-teacher.v3";
constexpr std::string_view kTerminationAuditSchema =
    "papersoccer.jacek-replay-search-termination-audit.v2";
constexpr std::string_view kHeader =
    "position_id\troot_group_id\tgroup_id\tsource\tsplit\twinner\tmover\tprefix";

struct Options {
  std::string model_path;
  std::string model_sha256;
  std::string campaign_id;
  std::uint32_t max_time_ms{60'000U};
  std::size_t max_tree_nodes{64'000U};
  std::size_t max_actions{250U};
  std::size_t max_partial_paths{50'000U};
  double exploration{0.5};
  double fpu{0.5};
  bool audit_terminations{};
};

struct PositionRecord {
  std::string position_id;
  std::string root_group_id;
  std::string group_id;
  std::string source;
  std::string split;
  int winner{-1};
  int mover{-1};
  std::vector<int> prefix_players;
  std::vector<std::string> prefix_actions;
  ps::GameState state;
};

struct Label {
  PositionRecord record;
  std::uint64_t seed{};
  ps::JacekReplayBfmSearchStats stats{};
  float teacher_value{};
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
          out << "\\u00" << hex[character >> 4U]
              << hex[character & 15U];
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
  if (raw.empty() || error != std::errc{} ||
      end != raw.data() + raw.size() || result == 0U) {
    throw std::invalid_argument(std::string(label) +
                                " must be a positive integer");
  }
  return result;
}

double parse_double(std::string_view raw, std::string_view label) {
  std::string owned(raw);
  char *end = nullptr;
  const double result = std::strtod(owned.c_str(), &end);
  if (owned.empty() || end != owned.c_str() + owned.size() ||
      !std::isfinite(result)) {
    throw std::invalid_argument(std::string(label) +
                                " must be a finite number");
  }
  return result;
}

std::string required_value(int &index, int argc, char **argv,
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
          << "usage: papersoccer_jacek_replay_search_teacher "
             "--model PATH --model-sha256 HEX --campaign-id ID "
             "[--tree-nodes N] [--time-ms N] [--max-actions N] "
             "[--max-partial-paths N] [--exploration C] [--fpu V] "
             "[--audit-terminations]\n"
             "stdin TSV header: "
          << kHeader << '\n';
      std::exit(0);
    }
    if (option == "--audit-terminations") {
      options.audit_terminations = true;
      continue;
    }
    const std::string value = required_value(index, argc, argv, option);
    if (option == "--model") {
      options.model_path = value;
    } else if (option == "--model-sha256") {
      options.model_sha256 = value;
    } else if (option == "--campaign-id") {
      options.campaign_id = value;
    } else if (option == "--tree-nodes" || option == "--nodes") {
      options.max_tree_nodes =
          parse_unsigned<std::size_t>(value, option);
    } else if (option == "--time-ms") {
      options.max_time_ms =
          parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--max-actions") {
      options.max_actions = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--max-partial-paths") {
      options.max_partial_paths =
          parse_unsigned<std::size_t>(value, option);
    } else if (option == "--exploration") {
      options.exploration = parse_double(value, option);
    } else if (option == "--fpu") {
      options.fpu = parse_double(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  require_text(options.model_path, "--model");
  require_text(options.campaign_id, "--campaign-id");
  if (!is_lower_sha256(options.model_sha256)) {
    throw std::invalid_argument(
        "--model-sha256 must be 64 lowercase hexadecimal characters");
  }
  if (options.max_tree_nodes < 2U ||
      options.max_tree_nodes > 1'000'000U || options.max_actions == 0U ||
      options.max_actions > 250U || options.max_partial_paths == 0U ||
      options.max_partial_paths > 50'000U || options.exploration < 0.0 ||
      options.fpu < -1.0 || options.fpu > 1.0) {
    throw std::invalid_argument("invalid search-teacher configuration");
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
    if (end == std::string_view::npos) {
      return result;
    }
    begin = end + 1U;
  }
}

ps::RulesConfig contest_rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

ps::Move decode_direction(ps::Point from, char direction) {
  static constexpr std::array<ps::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
  }};
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("prefix contains an invalid direction");
  }
  const ps::Point delta = deltas[static_cast<std::size_t>(direction - '0')];
  return ps::Move{{from.x + delta.x, from.y + delta.y}};
}

void apply_encoded_turn(ps::GameState &state, std::string_view action) {
  if (action.empty()) {
    throw std::invalid_argument("prefix contains an empty turn");
  }
  ps::GameState next = state;
  const ps::Player mover = next.to_move;
  for (const char direction : action) {
    if (ps::is_terminal(next) || next.to_move != mover) {
      throw std::invalid_argument(
          "prefix turn continues after its complete-turn boundary");
    }
    next = ps::apply_move(next, decode_direction(next.ball, direction));
  }
  if (!ps::is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("prefix turn stops during a rebound");
  }
  state = std::move(next);
}

int parse_player(std::string_view value, std::string_view label) {
  if (value == "0") {
    return 0;
  }
  if (value == "1") {
    return 1;
  }
  throw std::invalid_argument(std::string(label) + " must be zero or one");
}

PositionRecord parse_record(std::string_view line) {
  const std::vector<std::string_view> fields = split(line, '\t');
  if (fields.size() != 8U) {
    throw std::invalid_argument(
        "position input must contain exactly eight TSV fields");
  }
  require_text(fields[0], "position_id");
  require_text(fields[1], "root_group_id");
  require_text(fields[2], "group_id");
  require_text(fields[3], "source");
  require_text(fields[4], "split");
  if (fields[4] != "train" && fields[4] != "validation" &&
      fields[4] != "test") {
    throw std::invalid_argument(
        "split must be train, validation, or test");
  }

  PositionRecord record;
  record.position_id = fields[0];
  record.root_group_id = fields[1];
  record.group_id = fields[2];
  record.source = fields[3];
  record.split = fields[4];
  record.winner = parse_player(fields[5], "winner");
  record.mover = parse_player(fields[6], "mover");
  record.state = ps::make_initial_state(contest_rules());
  if (!fields[7].empty()) {
    const std::vector<std::string_view> actions = split(fields[7], '/');
    for (std::size_t turn = 0; turn < actions.size(); ++turn) {
      if (actions[turn].empty()) {
        throw std::invalid_argument("prefix contains an empty turn");
      }
      const int prefix_mover = player_id(record.state.to_move);
      try {
        apply_encoded_turn(record.state, actions[turn]);
      } catch (const std::exception &error) {
        throw std::invalid_argument("prefix turn " + std::to_string(turn) +
                                    ": " + error.what());
      }
      record.prefix_players.push_back(prefix_mover);
      record.prefix_actions.emplace_back(actions[turn]);
    }
  }
  if (ps::is_terminal(record.state)) {
    throw std::invalid_argument("prefix resolves to a terminal position");
  }
  if (player_id(record.state.to_move) != record.mover) {
    throw std::invalid_argument(
        "declared mover does not match the replayed prefix");
  }
  return record;
}

std::vector<PositionRecord> read_records(std::istream &input) {
  std::string line;
  if (!std::getline(input, line) || line != kHeader) {
    throw std::invalid_argument(
        "position input must begin with the exact TSV header");
  }
  std::vector<PositionRecord> records;
  std::set<std::string> position_ids;
  std::size_t line_number = 1U;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty()) {
      throw std::invalid_argument("line " + std::to_string(line_number) +
                                  ": empty rows are not allowed");
    }
    try {
      PositionRecord record = parse_record(line);
      if (!position_ids.insert(record.position_id).second) {
        throw std::invalid_argument("duplicate position_id");
      }
      records.push_back(std::move(record));
    } catch (const std::exception &error) {
      throw std::invalid_argument("line " + std::to_string(line_number) +
                                  ": " + error.what());
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("could not read complete position input");
  }
  if (records.empty()) {
    throw std::invalid_argument("position input contains no data rows");
  }
  return records;
}

void write_prefix(std::ostream &out, const PositionRecord &record) {
  out << '[';
  for (std::size_t index = 0; index < record.prefix_actions.size(); ++index) {
    if (index != 0U) {
      out << ',';
    }
    out << "{\"player_id\":" << record.prefix_players[index]
        << ",\"action\":" << json_string(record.prefix_actions[index])
        << '}';
  }
  out << ']';
}

void write_bool(std::ostream &out, bool value) {
  out << (value ? "true" : "false");
}

void write_search_stats(std::ostream &out,
                        const ps::JacekReplayBfmSearchStats &stats) {
  out << "{\"expansions\":" << stats.expansions
      << ",\"generated_actions\":" << stats.generated_actions
      << ",\"retained_actions\":" << stats.retained_actions
      << ",\"neural_evaluations\":" << stats.neural_evaluations
      << ",\"visits\":" << stats.visits
      << ",\"completed_actions\":" << stats.completed_actions
      << ",\"duplicate_boundaries\":" << stats.duplicate_boundaries
      << ",\"partial_paths\":" << stats.partial_paths
      << ",\"fifo_extractions\":" << stats.fifo_extractions
      << ",\"lifo_extractions\":" << stats.lifo_extractions
      << ",\"tactical_proofs\":" << stats.tactical_proofs
      << ",\"tactical_solutions\":" << stats.tactical_solutions
      << ",\"truncations\":" << stats.truncations
      << ",\"generation_action_cap_stops\":"
      << stats.generation_action_cap_stops
      << ",\"generation_partial_cap_stops\":"
      << stats.generation_partial_cap_stops
      << ",\"generation_deadline_stops\":"
      << stats.generation_deadline_stops
      << ",\"materialization_deadline_stops\":"
      << stats.materialization_deadline_stops
      << ",\"generation_queue_drops\":"
      << stats.generation_queue_drops
      << ",\"generation_retention_drops\":"
      << stats.generation_retention_drops
      << ",\"generation_boundary_replacements\":"
      << stats.generation_boundary_replacements
      << ",\"generation_tactical_shortcuts\":"
      << stats.generation_tactical_shortcuts
      << ",\"generation_fallbacks\":" << stats.generation_fallbacks
      << ",\"generation_frontier_resumptions\":"
      << stats.generation_frontier_resumptions
      << ",\"generation_zero_action_resumptions\":"
      << stats.generation_zero_action_resumptions
      << ",\"generation_max_frontier_depth\":"
      << stats.generation_max_frontier_depth
      << ",\"progressive_widenings\":" << stats.progressive_widenings
      << ",\"closed_unsolved_nodes\":" << stats.closed_unsolved_nodes
      << ",\"closed_unsolved_nonexhaustive_nodes\":"
      << stats.closed_unsolved_nonexhaustive_nodes
      << ",\"open_unexpanded_nodes\":" << stats.open_unexpanded_nodes
      << ",\"implicit_action_frontiers\":"
      << stats.implicit_action_frontiers
      << ",\"max_open_children\":" << stats.max_open_children
      << ",\"tree_nodes\":" << stats.tree_nodes
      << ",\"max_complete_turn_depth\":"
      << stats.max_complete_turn_depth << ",\"deadline_reached\":";
  write_bool(out, stats.deadline_reached);
  out << ",\"tree_cap_reached\":";
  write_bool(out, stats.tree_cap_reached);
  out << ",\"termination_reason\":"
      << json_string(ps::jacek_replay_bfm_search_termination_name(
             stats.termination))
      << '}';
}

void write_termination_audit(
    std::ostream &out, const Options &options, const PositionRecord &record,
    std::uint64_t seed, const ps::JacekReplayBfmSearchStats &stats,
    std::string_view model_sha256) {
  bool teacher_contract_valid = false;
  std::optional<float> teacher_value;
  std::string contract_error;
  try {
    teacher_value = direct_teacher_value(stats, record.state.to_move);
    teacher_contract_valid = true;
  } catch (const std::exception &error) {
    contract_error = error.what();
  }
  const bool premature = !teacher_contract_valid;

  out << "{\"schema\":" << json_string(kTerminationAuditSchema)
      << ",\"campaign_id\":" << json_string(options.campaign_id)
      << ",\"position_id\":" << json_string(record.position_id)
      << ",\"model_sha256\":" << json_string(model_sha256)
      << ",\"seed\":" << seed << ",\"termination_reason\":"
      << json_string(ps::jacek_replay_bfm_search_termination_name(
             stats.termination))
      << ",\"premature\":";
  write_bool(out, premature);
  out << ",\"teacher_contract_valid\":";
  write_bool(out, teacher_contract_valid);
  out << ",\"teacher_contract_error\":";
  if (contract_error.empty()) {
    out << "null";
  } else {
    out << json_string(contract_error);
  }
  out << ",\"teacher_value\":";
  if (teacher_value.has_value()) {
    out << std::setprecision(std::numeric_limits<float>::max_digits10)
        << *teacher_value;
  } else {
    out << "null";
  }
  out << ",\"root_solved\":";
  write_bool(out, stats.root_solved);
  out << ",\"search_stats\":";
  write_search_stats(out, stats);
  out << "}\n";
}

void write_label(std::ostream &out, const Options &options,
                 const Label &label, std::string_view model_sha256) {
  const PositionRecord &record = label.record;
  const ps::JacekReplayBfmSearchStats &stats = label.stats;
  out << "{\"schema\":" << json_string(kSchema)
      << ",\"campaign_id\":" << json_string(options.campaign_id)
      << ",\"position_id\":" << json_string(record.position_id)
      << ",\"root_group_id\":" << json_string(record.root_group_id)
      << ",\"group_id\":" << json_string(record.group_id)
      << ",\"source\":" << json_string(record.source)
      << ",\"split\":" << json_string(record.split)
      << ",\"winner\":" << record.winner << ",\"prefix\":";
  write_prefix(out, record);
  out << ",\"mover\":" << record.mover
      << ",\"teacher\":{\"kind\":\"jacek_replay_bfm_search\""
      << ",\"source_sha256\":"
      << json_string(PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256)
      << ",\"model_sha256\":" << json_string(model_sha256)
      << ",\"feature_schema\":"
      << json_string(ps::JacekReplayBfmBot::feature_schema())
      << ",\"feature_schema_sha256\":"
      << json_string(ps::JacekReplayBfmBot::feature_schema_sha256()) << '}'
      << ",\"search_config\":{\"seed\":" << label.seed
      << ",\"max_time_ms\":" << options.max_time_ms
      << ",\"max_tree_nodes\":" << options.max_tree_nodes
      << ",\"max_actions\":" << options.max_actions
      << ",\"max_partial_paths\":" << options.max_partial_paths
      << ",\"exploration\":"
      << std::setprecision(std::numeric_limits<double>::max_digits10)
      << options.exploration << ",\"fpu\":" << options.fpu << '}'
      << ",\"search_stats\":";
  write_search_stats(out, stats);
  out << ",\"teacher_value\":"
      << std::setprecision(std::numeric_limits<float>::max_digits10)
      << label.teacher_value << ",\"root_solved\":";
  write_bool(out, stats.root_solved);
  out << ",\"proven_winner\":";
  if (stats.proven_winner.has_value()) {
    out << player_id(*stats.proven_winner);
  } else {
    out << "null";
  }
  out << ",\"weight\":1.0}\n";
}

std::vector<Label> analyze(const Options &options,
                           std::vector<PositionRecord> records,
                           const ps::JacekReplayBfmBot &bot) {
  std::vector<Label> labels;
  labels.reserve(records.size());
  for (PositionRecord &record : records) {
    const std::uint64_t seed = derive_search_seed(
        options.campaign_id, record.position_id, options.max_tree_nodes);
    const ps::JacekReplayBfmSearchStats stats =
        bot.analyze_position(record.state, seed);
    const float value = direct_teacher_value(stats, record.state.to_move);
    labels.push_back(Label{std::move(record), seed, stats, value});
  }
  return labels;
}

}  // namespace

std::uint64_t derive_search_seed(std::string_view campaign_id,
                                 std::string_view position_id,
                                 std::size_t max_tree_nodes) {
  require_text(campaign_id, "campaign_id");
  require_text(position_id, "position_id");
  std::string material;
  material.reserve(campaign_id.size() + position_id.size() + 32U);
  material.append(campaign_id);
  material.push_back('\0');
  material.append(position_id);
  material.push_back('\0');
  material.append(std::to_string(max_tree_nodes));
  const std::span<const std::uint8_t> bytes(
      reinterpret_cast<const std::uint8_t *>(material.data()),
      material.size());
  const std::string digest = ps::detail::replay_bfm_sha256_hex(bytes);
  std::uint64_t seed{};
  const auto [end, error] =
      std::from_chars(digest.data(), digest.data() + 16U, seed, 16);
  if (error != std::errc{} || end != digest.data() + 16U) {
    throw std::logic_error("could not derive the search seed");
  }
  return seed;
}

void validate_model_identity(std::string_view actual_sha256,
                             std::string_view expected_sha256) {
  if (!is_lower_sha256(actual_sha256) ||
      !is_lower_sha256(expected_sha256)) {
    throw std::invalid_argument(
        "model identities must be lowercase SHA-256 values");
  }
  if (actual_sha256 != expected_sha256) {
    throw std::invalid_argument(
        "loaded model SHA-256 does not match --model-sha256");
  }
}

float direct_teacher_value(const ps::JacekReplayBfmSearchStats &stats,
                           ps::Player mover) {
  if (stats.deadline_reached ||
      stats.termination == ps::JacekReplayBfmSearchTermination::Deadline ||
      stats.generation_deadline_stops != 0U ||
      stats.materialization_deadline_stops != 0U) {
    throw std::runtime_error("search teacher reached its deadline");
  }
  if (stats.closed_unsolved_nodes != 0U ||
      stats.closed_unsolved_nonexhaustive_nodes != 0U) {
    throw std::runtime_error("search teacher closed an unsolved frontier");
  }
  if (stats.generation_queue_drops != 0U) {
    throw std::runtime_error("search teacher dropped a partial frontier");
  }
  if (stats.max_complete_turn_depth == 0U) {
    throw std::runtime_error(
        "search teacher did not complete a root search depth");
  }
  if (!std::isfinite(stats.root_value)) {
    throw std::runtime_error("search teacher produced a non-finite value");
  }
  if (stats.root_solved) {
    if (stats.termination !=
        ps::JacekReplayBfmSearchTermination::RootSolved) {
      throw std::runtime_error(
          "solved search teacher root has an inconsistent termination");
    }
    if (!stats.proven_winner.has_value()) {
      throw std::runtime_error(
          "solved search teacher root is missing its proven winner");
    }
    const float expected = *stats.proven_winner == mover ? 1.0F : -1.0F;
    if ((expected > 0.0F && stats.root_value <= 0.0F) ||
        (expected < 0.0F && stats.root_value >= 0.0F)) {
      throw std::runtime_error(
          "solved search teacher value contradicts its proven winner");
    }
    return expected;
  }
  if (!stats.tree_cap_reached ||
      stats.termination !=
          ps::JacekReplayBfmSearchTermination::FixedWorkCap) {
    throw std::runtime_error(
        "search teacher termination " +
        std::string(ps::jacek_replay_bfm_search_termination_name(
            stats.termination)) +
        " is not a proof or completed fixed-work cap");
  }
  if (stats.proven_winner.has_value()) {
    throw std::runtime_error(
        "unsolved search teacher root has a proven winner");
  }
  if (stats.root_value < -1.0F || stats.root_value > 1.0F) {
    throw std::runtime_error(
        "unsolved search teacher value is outside [-1,1]");
  }
  return stats.root_value;
}

int run(int argc, char **argv, std::istream &input, std::ostream &output) {
  if (!is_lower_sha256(
          PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256)) {
    throw std::logic_error(
        "search-teacher source closure identity is invalid");
  }
  const Options options = parse_options(argc, argv);
  ps::JacekReplayBfmConfig config;
  config.model_path = options.model_path;
  config.max_time_ms = options.max_time_ms;
  config.max_tree_nodes = options.max_tree_nodes;
  config.max_actions = options.max_actions;
  config.max_partial_paths = options.max_partial_paths;
  config.exploration = options.exploration;
  config.fpu = options.fpu;
  ps::JacekReplayBfmBot bot(std::move(config));
  validate_model_identity(bot.model_sha256(), options.model_sha256);
  std::vector<PositionRecord> records = read_records(input);
  if (options.audit_terminations) {
    for (const PositionRecord &record : records) {
      const std::uint64_t seed = derive_search_seed(
          options.campaign_id, record.position_id, options.max_tree_nodes);
      const ps::JacekReplayBfmSearchStats stats =
          bot.analyze_position(record.state, seed);
      write_termination_audit(
          output, options, record, seed, stats, bot.model_sha256());
    }
    output.flush();
    if (!output) {
      throw std::runtime_error(
          "could not write complete search-termination audit");
    }
    return 0;
  }
  const std::vector<Label> labels =
      analyze(options, std::move(records), bot);
  for (const Label &label : labels) {
    write_label(output, options, label, bot.model_sha256());
  }
  output.flush();
  if (!output) {
    throw std::runtime_error(
        "could not write complete search-teacher output");
  }
  return 0;
}

}  // namespace papersoccer::jacek_replay_search_teacher

#ifndef PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_NO_MAIN
int main(int argc, char **argv) {
  try {
    return papersoccer::jacek_replay_search_teacher::run(
        argc, argv, std::cin, std::cout);
  } catch (const std::exception &error) {
    std::cerr << "jacek replay search teacher: " << error.what() << '\n';
    return 2;
  }
}
#endif
