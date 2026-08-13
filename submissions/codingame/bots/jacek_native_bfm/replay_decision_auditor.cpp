#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include "bot.cpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace papersoccer::jacek_native_replay_audit {

namespace native = papersoccer::jacek_native_bfm;

inline constexpr std::string_view kSchemaVersion =
    "jacek-native-decision-audit-v4";
inline constexpr std::string_view kRuntimeSchema =
    "papersoccer.jacek-native-runtime-model/v1";
inline constexpr std::string_view kInputHeader =
    "game_id\tcandidate_player\twinner\tturns";
inline constexpr std::size_t kMaximumGames = 4'096;
inline constexpr std::size_t kMaximumGameIdBytes = 128;
inline constexpr std::size_t kMaximumTranscriptBytes = 65'536;
inline constexpr std::size_t kMaximumTurns = 1'024;
inline constexpr std::size_t kMaximumActionBytes = 1'024;
inline constexpr std::size_t kMaximumProvenanceEntries = 64;
inline constexpr std::size_t kMaximumProvenanceKeyBytes = 64;
inline constexpr std::size_t kMaximumProvenanceValueBytes = 512;

enum class OutputFormat { JsonLines, Tsv };
enum class AuditMode { FixedWork, Clock };

using ProvenanceMetadata = std::vector<std::pair<std::string, std::string>>;

struct AuditConfig {
  AuditMode mode{AuditMode::FixedWork};
  std::size_t fixed_work{30'000};
  std::size_t max_actions{native::kMaximumActions};
  std::size_t max_partial_paths{native::kProductionPartialPaths};
  std::uint64_t max_expansions{native::kMaximumExpansions};
  std::size_t root_reply_width{};
  double supported_advance_penalty{};
  double exploration{native::kExplorationConstant};
  double first_play_urgency{native::kFirstPlayUrgency};
  std::uint32_t first_time_ms{};
  std::uint32_t later_time_ms{};
  OutputFormat output_format{OutputFormat::JsonLines};
  std::optional<std::string> input_path{};
  std::optional<std::string> checkpoint_path{};
  std::optional<std::string> selected_prefixes_path{};
  std::optional<std::size_t> max_own_decisions_per_game{};
};

struct RuntimeIdentity {
  std::string runtime_sha256;
  std::string model_sha256;
  std::string packed_sha256;
};

struct LoadedRuntime {
  std::string model_schema;
  std::string feature_schema;
  std::string packed;
  float scale_one{};
  float scale_two{};
  float scale_three{};
  RuntimeIdentity identity;
};

struct SelectedPrefix {
  std::string game_id;
  std::size_t turn_index{};
  std::string state_id;
  bool matched{};
};

struct GameRecord {
  std::string game_id;
  int candidate_player{};
  int winner{};
  std::vector<std::string> actions;
  ProvenanceMetadata provenance;
};

struct ActionDiagnostic {
  std::string action;
  int exact_retained_ordinal{-1};
  int boundary_retained_ordinal{-1};
  std::string retained_action;
  std::string tactical_class{"not-retained"};
  std::optional<float> initial_neural_value;
  std::optional<float> initial_action_value;
  int initial_rank{-1};
  std::optional<float> final_backed_value;
  std::optional<std::uint32_t> final_visits;
  std::optional<std::uint32_t> final_selection_visits;
  std::string final_tactical_class{"not-searched"};
  bool exact_reply_refuted{};
  bool final_solved{};
  bool final_proven_win{};
  std::optional<int> start_progress;
  std::optional<int> endpoint_progress;
  bool supported_advance_eligible{};
  double supported_advance_penalty{};
  std::optional<double> final_unpenalized_score;
  std::optional<double> final_penalized_score;
};

struct RootDiagnostics {
  std::size_t retained_actions{};
  std::uint64_t explored_partial_paths{};
  std::uint64_t proof_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t duplicate_boundaries{};
  std::uint64_t fifo_extractions{};
  std::uint64_t lifo_extractions{};
  std::uint64_t tactical_actions{};
  std::uint8_t proof_classes{};
  std::uint64_t truncations{};
  std::uint32_t maximum_deque_size{};
  bool deadline_reached{};
  bool proof_truncated{};
  bool exhaustive{};
  std::string initial_best_action;
  float initial_best_value{};
};

struct SearchDecision {
  std::string action;
  float value{};
  bool solved{};
  std::optional<int> solved_winner;
  double elapsed_ms{};
  std::size_t root_actions{};
  std::vector<std::string> exact_reply_refuted_actions;
  native::SearchStats stats{};
  struct RootAction {
    std::string encoded;
    float value{};
    float initial_value{};
    std::uint32_t visits{};
    std::uint32_t selection_visits{};
    std::string tactical_class;
    bool solved{};
    bool proven_win{};
    bool exact_reply_refuted{};
    int start_progress{};
    int endpoint_progress{};
    bool supported_advance_eligible{};
    double supported_advance_penalty{};
    double unpenalized_final_score{};
    double penalized_final_score{};
  };
  std::vector<RootAction> root_action_diagnostics;
};

struct DecisionAudit {
  std::string game_id;
  std::string state_id;
  std::string transcript_prefix;
  std::size_t turn_index{};
  std::size_t own_decision_index{};
  int candidate_player{};
  int winner{};
  std::uint32_t time_limit_ms{};
  std::size_t pre_action_used_edges{};
  ProvenanceMetadata provenance;
  RootDiagnostics root;
  ActionDiagnostic actual;
  ActionDiagnostic chosen;
  SearchDecision search;
  std::string classification;
  std::string classification_reason;
  RuntimeIdentity runtime;
};

RulesConfig codingame_rules() {
  return RulesConfig{8, 10, GoalRule::OwnGoalsAllowed,
                     BlockedRule::MoverLoses};
}

int player_id(Player player) noexcept {
  return player == Player::One ? 0 : 1;
}

std::vector<std::string_view> split_preserving_empty(std::string_view value,
                                                     char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = value.find(separator, begin);
    result.push_back(value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return result;
    begin = end + 1U;
  }
}

template <typename Integer>
Integer parse_integer(std::string_view raw, std::string_view label) {
  Integer value{};
  const char *const begin = raw.data();
  const char *const end = begin + raw.size();
  const auto [parsed, error] = std::from_chars(begin, end, value);
  if (raw.empty() || error != std::errc{} || parsed != end) {
    throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                std::string(raw));
  }
  return value;
}

double parse_double(std::string_view raw, std::string_view label) {
  std::string owned(raw);
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(owned, &consumed);
  } catch (const std::exception &) {
    throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                owned);
  }
  if (consumed != owned.size() || !std::isfinite(value)) {
    throw std::invalid_argument("invalid " + std::string(label) + ": " +
                                owned);
  }
  return value;
}

bool valid_sha256(std::string_view value) noexcept {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

LoadedRuntime parse_runtime(std::string_view raw) {
  if (raw.empty() || raw.back() != '\n' || raw.find('\r') != raw.npos ||
      raw.find('\0') != raw.npos) {
    throw std::invalid_argument("runtime checkpoint is not canonical text");
  }
  const std::vector<std::string_view> lines =
      split_preserving_empty(raw, '\n');
  if (lines.size() != 8 || !lines.back().empty() ||
      lines[0] != kRuntimeSchema || !valid_sha256(lines[3]) ||
      !valid_sha256(lines[4])) {
    throw std::invalid_argument("runtime checkpoint metadata is invalid");
  }
  std::istringstream scales{std::string(lines[5])};
  LoadedRuntime runtime;
  if (!(scales >> runtime.scale_one >> runtime.scale_two >>
        runtime.scale_three)) {
    throw std::invalid_argument("runtime checkpoint scales are invalid");
  }
  scales >> std::ws;
  if (!scales.eof() || !std::isfinite(runtime.scale_one) ||
      !std::isfinite(runtime.scale_two) ||
      !std::isfinite(runtime.scale_three) || runtime.scale_one <= 0.0F ||
      runtime.scale_two <= 0.0F || runtime.scale_three <= 0.0F) {
    throw std::invalid_argument("runtime checkpoint scales are invalid");
  }
  runtime.model_schema = lines[1];
  runtime.feature_schema = lines[2];
  runtime.packed = lines[6];
  runtime.identity.model_sha256 = lines[3];
  runtime.identity.packed_sha256 = lines[4];
  const std::vector<std::uint8_t> bytes(raw.begin(), raw.end());
  runtime.identity.runtime_sha256 =
      native::sha256_hex(std::span<const std::uint8_t>(bytes));
  const native::QuantizedModel verified(
      runtime.packed, runtime.scale_one, runtime.scale_two,
      runtime.scale_three, runtime.model_schema, runtime.feature_schema,
      runtime.identity.model_sha256, runtime.identity.model_sha256, false,
      runtime.identity.packed_sha256);
  if (verified.model_sha256() != runtime.identity.model_sha256 ||
      verified.packed_sha256() != runtime.identity.packed_sha256) {
    throw std::invalid_argument("runtime checkpoint hashes do not match payload");
  }
  return runtime;
}

std::string default_runtime_text() {
  std::ostringstream output;
  output << kRuntimeSchema << '\n' << jacek_native_model::kModelSchema << '\n'
         << jacek_native_model::kFeatureSchema << '\n'
         << jacek_native_model::kModelSha256 << '\n'
         << jacek_native_model::kPackedSha256 << '\n' << std::setprecision(9)
         << jacek_native_model::kScale1 << ' '
         << jacek_native_model::kScale2 << ' '
         << jacek_native_model::kScale3 << '\n'
         << jacek_native_model::kPackedWeights << '\n';
  return output.str();
}

LoadedRuntime default_runtime() { return parse_runtime(default_runtime_text()); }

LoadedRuntime load_runtime(const std::optional<std::string> &path) {
  if (!path.has_value()) return default_runtime();
  std::ifstream input(*path, std::ios::binary);
  if (!input) {
    throw std::invalid_argument("cannot open runtime checkpoint: " + *path);
  }
  const std::string raw{std::istreambuf_iterator<char>(input),
                        std::istreambuf_iterator<char>()};
  return parse_runtime(raw);
}

native::QuantizedModel make_runtime_model(const LoadedRuntime &runtime) {
  return native::QuantizedModel(
      runtime.packed, runtime.scale_one, runtime.scale_two,
      runtime.scale_three, runtime.model_schema, runtime.feature_schema,
      runtime.identity.model_sha256, runtime.identity.model_sha256, false,
      runtime.identity.packed_sha256);
}

bool safe_game_id(std::string_view value) noexcept {
  if (value.empty() || value.size() > kMaximumGameIdBytes) return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '-' ||
           character == '_' || character == '.' || character == ':';
  });
}

bool safe_provenance_key(std::string_view value) noexcept {
  if (value.empty() || value.size() > kMaximumProvenanceKeyBytes) return false;
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '_' ||
           character == '-' || character == '.';
  });
}

void parse_provenance_comment(
    std::string_view line, std::size_t line_number,
    ProvenanceMetadata &provenance,
    std::unordered_set<std::string> &provenance_keys) {
  line.remove_prefix(1U);
  while (!line.empty() && line.front() == ' ') line.remove_prefix(1U);
  const std::size_t separator = line.find('=');
  if (separator == std::string_view::npos) return;
  const std::string_view key = line.substr(0, separator);
  const std::string_view value = line.substr(separator + 1U);
  if (!safe_provenance_key(key) || value.empty() ||
      value.size() > kMaximumProvenanceValueBytes ||
      !std::all_of(value.begin(), value.end(), [](char character) {
        return character >= 0x20 && character <= 0x7e;
      })) {
    throw std::invalid_argument(
        "line " + std::to_string(line_number) +
        " has invalid provenance; expected # safe_key=printable_value");
  }
  if (provenance.size() >= kMaximumProvenanceEntries) {
    throw std::invalid_argument("input exceeds the provenance entry limit");
  }
  if (!provenance_keys.insert(std::string(key)).second) {
    throw std::invalid_argument("duplicate provenance key: " +
                                std::string(key));
  }
  provenance.emplace_back(key, value);
}

GameRecord parse_record(std::string_view line, std::size_t line_number) {
  const std::vector<std::string_view> fields =
      split_preserving_empty(line, '\t');
  if (fields.size() != 4) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " must contain exactly four TSV fields");
  }
  if (!safe_game_id(fields[0])) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " has an unsafe or empty game_id");
  }
  const int candidate_player = parse_integer<int>(fields[1], "candidate_player");
  const int winner = parse_integer<int>(fields[2], "winner");
  if ((candidate_player != 0 && candidate_player != 1) ||
      (winner != 0 && winner != 1)) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " requires player identities in {0,1}");
  }
  if (fields[3].empty() || fields[3].size() > kMaximumTranscriptBytes) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " has an empty or overlong transcript");
  }
  const std::vector<std::string_view> raw_actions =
      split_preserving_empty(fields[3], '/');
  if (raw_actions.empty() || raw_actions.size() > kMaximumTurns) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " has an invalid turn count");
  }
  GameRecord record{std::string(fields[0]), candidate_player, winner, {}, {}};
  record.actions.reserve(raw_actions.size());
  for (const std::string_view action : raw_actions) {
    if (action.empty() || action.size() > kMaximumActionBytes ||
        !std::all_of(action.begin(), action.end(), [](char direction) {
          return direction >= '0' && direction <= '7';
        })) {
      throw std::invalid_argument("line " + std::to_string(line_number) +
                                  " has an invalid complete-turn action");
    }
    record.actions.emplace_back(action);
  }
  return record;
}

std::vector<GameRecord> read_records(std::istream &input) {
  std::vector<GameRecord> records;
  std::unordered_set<std::string> game_ids;
  ProvenanceMetadata provenance;
  std::unordered_set<std::string> provenance_keys;
  std::string line;
  std::size_t line_number = 0;
  bool saw_header = false;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line.front() == '#') {
      if (!saw_header) {
        parse_provenance_comment(line, line_number, provenance,
                                 provenance_keys);
      }
      continue;
    }
    if (!saw_header) {
      if (line != kInputHeader) {
        throw std::invalid_argument("first data line must be: " +
                                    std::string(kInputHeader));
      }
      saw_header = true;
      continue;
    }
    if (records.size() >= kMaximumGames) {
      throw std::invalid_argument("input exceeds the maximum game count");
    }
    GameRecord record = parse_record(line, line_number);
    record.provenance = provenance;
    if (!game_ids.insert(record.game_id).second) {
      throw std::invalid_argument("duplicate game_id: " + record.game_id);
    }
    records.push_back(std::move(record));
  }
  if (!saw_header) throw std::invalid_argument("missing TSV input header");
  if (records.empty()) throw std::invalid_argument("input contains no games");
  return records;
}

std::vector<SelectedPrefix> read_selected_prefixes(std::istream &input) {
  constexpr std::string_view header = "game_id\tturn_index\tstate_id";
  std::vector<SelectedPrefix> result;
  std::unordered_set<std::string> keys;
  std::string line;
  if (!std::getline(input, line) || line != header) {
    throw std::invalid_argument("selected-prefix manifest header must be: " +
                                std::string(header));
  }
  if (line.find('\r') != std::string::npos) {
    throw std::invalid_argument("selected-prefix manifest must use LF");
  }
  while (std::getline(input, line)) {
    if (line.empty() || line.find('\r') != std::string::npos) {
      throw std::invalid_argument(
          "selected-prefix manifest contains a blank or CR line");
    }
    if (result.size() >= kMaximumGames) {
      throw std::invalid_argument("selected-prefix manifest is too large");
    }
    const auto fields = split_preserving_empty(line, '\t');
    if (fields.size() != 3 || !safe_game_id(fields[0]) ||
        fields[2].size() != 24 || fields[2].substr(0, 8) != "fnv1a64:" ||
        !std::all_of(fields[2].begin() + 8, fields[2].end(), [](char value) {
          return (value >= '0' && value <= '9') ||
                 (value >= 'a' && value <= 'f');
        })) {
      throw std::invalid_argument("selected-prefix manifest row is malformed");
    }
    const std::size_t turn =
        parse_integer<std::size_t>(fields[1], "selected turn index");
    if (turn >= kMaximumTurns) {
      throw std::invalid_argument("selected turn index is out of range");
    }
    std::string key(fields[0]);
    key.push_back('\t');
    key.append(fields[1]);
    if (!keys.insert(key).second) {
      throw std::invalid_argument("duplicate selected-prefix key: " + key);
    }
    result.push_back(
        {std::string(fields[0]), turn, std::string(fields[2]), false});
  }
  if (result.empty()) {
    throw std::invalid_argument("selected-prefix manifest is empty");
  }
  return result;
}

void validate_record(const GameRecord &record) {
  GameState state = make_initial_state(codingame_rules());
  for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
    if (is_terminal(state)) {
      throw std::invalid_argument("game " + record.game_id +
                                  " contains a turn after terminal state");
    }
    try {
      native::apply_encoded_turn(state, record.actions[turn]);
    } catch (const std::exception &error) {
      throw std::invalid_argument("game " + record.game_id + " turn " +
                                  std::to_string(turn) + " is invalid: " +
                                  error.what());
    }
  }
  if (!is_terminal(state)) {
    throw std::invalid_argument("game " + record.game_id +
                                " transcript is not terminal");
  }
  const std::optional<Player> actual_winner = winner(state);
  if (!actual_winner.has_value() || player_id(*actual_winner) != record.winner) {
    throw std::invalid_argument("game " + record.game_id +
                                " winner does not match replay");
  }
}

void validate_records(const std::vector<GameRecord> &records) {
  for (const GameRecord &record : records) validate_record(record);
}

class Fnv64 {
 public:
  void add_byte(std::uint8_t value) noexcept {
    value_ ^= value;
    value_ *= 1'099'511'628'211ULL;
  }

  template <typename Integer>
  void add_integer(Integer value) noexcept {
    const std::uint64_t bits = static_cast<std::uint64_t>(value);
    for (unsigned shift = 0; shift < 64; shift += 8) {
      add_byte(static_cast<std::uint8_t>(bits >> shift));
    }
  }

  std::uint64_t value() const noexcept { return value_; }

 private:
  std::uint64_t value_{14'695'981'039'346'656'037ULL};
};

std::string state_identity(const GameState &state) {
  Fnv64 hash;
  hash.add_integer(state.config.width);
  hash.add_integer(state.config.height);
  hash.add_integer(static_cast<int>(state.config.goal_rule));
  hash.add_integer(static_cast<int>(state.config.blocked_rule));
  hash.add_integer(state.ball.x);
  hash.add_integer(state.ball.y);
  hash.add_integer(static_cast<int>(state.to_move));
  hash.add_integer(static_cast<int>(state.status));
  std::vector<Segment> segments(state.used_segments.begin(),
                                state.used_segments.end());
  std::sort(segments.begin(), segments.end(), [](const Segment &left,
                                                 const Segment &right) {
    if (left.a == right.a) return left.b < right.b;
    return left.a < right.a;
  });
  hash.add_integer(segments.size());
  for (const Segment &segment : segments) {
    hash.add_integer(segment.a.x);
    hash.add_integer(segment.a.y);
    hash.add_integer(segment.b.x);
    hash.add_integer(segment.b.y);
  }
  std::vector<std::pair<Point, int>> visits(state.visit_count.begin(),
                                             state.visit_count.end());
  std::sort(visits.begin(), visits.end(), [](const auto &left,
                                             const auto &right) {
    return left.first < right.first;
  });
  hash.add_integer(visits.size());
  for (const auto &[point, count] : visits) {
    hash.add_integer(point.x);
    hash.add_integer(point.y);
    hash.add_integer(count);
  }
  std::ostringstream output;
  output << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
         << hash.value();
  return output.str();
}

std::uint32_t decision_time_limit(const AuditConfig &config,
                                  std::size_t own_decision_index) noexcept {
  if (config.mode != AuditMode::Clock) return 0;
  return own_decision_index == 0 ? config.first_time_ms : config.later_time_ms;
}

std::string tactical_class_name(native::TacticalClass tactical) {
  switch (tactical) {
    case native::TacticalClass::ImmediateGoal:
      return "immediate-goal";
    case native::TacticalClass::ForcedCutoff:
      return "forced-cutoff";
    case native::TacticalClass::SafeHandoff:
      return "safe-handoff";
    case native::TacticalClass::OpponentImmediateGoal:
      return "opponent-immediate-goal";
    case native::TacticalClass::OwnGoal:
      return "own-goal";
  }
  throw std::logic_error("unknown tactical class");
}

int tactical_order(std::string_view tactical) noexcept {
  if (tactical == "immediate-goal") return 0;
  if (tactical == "forced-cutoff") return 1;
  if (tactical == "safe-handoff") return 2;
  if (tactical == "opponent-immediate-goal") return 3;
  if (tactical == "own-goal") return 4;
  return 5;
}

struct RankedAction {
  std::size_t retained_ordinal{};
  float action_value{};
  std::optional<float> neural_value;
  std::size_t rank{};
};

struct DiagnosticGeneration {
  std::shared_ptr<const native::SearchTopology> topology;
  native::GenerationResult generated;
  std::vector<RankedAction> ranked;
  RootDiagnostics root;
};

float exact_initial_action_value(const native::CompleteTurnAction &action,
                                 Player root_mover) {
  const Player child_perspective = opponent(root_mover);
  if (action.result.is_terminal()) {
    return -native::terminal_value(*action.result.winner(), child_perspective,
                                   1U);
  }
  if (action.tactical == native::TacticalClass::ForcedCutoff) {
    return -native::terminal_value(opponent(child_perspective),
                                   child_perspective, 2U);
  }
  if (action.tactical == native::TacticalClass::OpponentImmediateGoal) {
    return -native::terminal_value(child_perspective, child_perspective, 2U);
  }
  throw std::logic_error("non-proven action requested an exact initial value");
}

DiagnosticGeneration generate_root_diagnostics(
    const GameState &state, const AuditConfig &config,
    const native::QuantizedModel &model) {
  DiagnosticGeneration diagnostics;
  diagnostics.topology =
      std::make_shared<native::SearchTopology>(state.config);
  const native::SearchPosition root(diagnostics.topology, state);
  native::GeneratorConfig generator_config;
  generator_config.max_actions =
      std::min(config.max_actions, config.fixed_work - 1U);
  generator_config.max_partial_paths = config.max_partial_paths;
  native::CompleteTurnGenerator generator(diagnostics.topology, root,
                                           generator_config);
  diagnostics.generated = generator.run();

  native::NeuralEvaluator evaluator(diagnostics.topology, model);
  const Player child_perspective = opponent(root.to_move());
  const native::PreparedEvaluation prepared =
      evaluator.prepare(root, child_perspective);
  diagnostics.ranked.reserve(diagnostics.generated.actions.size());
  for (std::size_t ordinal = 0;
       ordinal < diagnostics.generated.actions.size(); ++ordinal) {
    const native::CompleteTurnAction &action =
        diagnostics.generated.actions[ordinal];
    RankedAction ranked;
    ranked.retained_ordinal = ordinal;
    if (action.result.is_terminal() ||
        action.tactical == native::TacticalClass::ForcedCutoff ||
        action.tactical == native::TacticalClass::OpponentImmediateGoal) {
      ranked.action_value = exact_initial_action_value(action, root.to_move());
    } else {
      const float child_value = evaluator.evaluate_child(prepared, action.result);
      ranked.neural_value = -child_value;
      ranked.action_value = *ranked.neural_value;
    }
    diagnostics.ranked.push_back(ranked);
  }
  std::stable_sort(diagnostics.ranked.begin(), diagnostics.ranked.end(),
                   [&](const RankedAction &left, const RankedAction &right) {
    if (left.action_value != right.action_value) {
      return left.action_value > right.action_value;
    }
    return diagnostics.generated.actions[left.retained_ordinal].encoded <
           diagnostics.generated.actions[right.retained_ordinal].encoded;
  });
  for (std::size_t index = 0; index < diagnostics.ranked.size(); ++index) {
    diagnostics.ranked[index].rank = index + 1U;
  }
  if (diagnostics.ranked.empty()) {
    throw std::logic_error("diagnostic root generation returned no actions");
  }
  const native::GeneratorStats &stats = diagnostics.generated.stats;
  diagnostics.root.retained_actions = diagnostics.generated.actions.size();
  diagnostics.root.explored_partial_paths = stats.explored_partial_paths;
  diagnostics.root.proof_paths = stats.proof_paths;
  diagnostics.root.completed_actions = stats.completed_actions;
  diagnostics.root.duplicate_boundaries = stats.duplicate_boundaries;
  diagnostics.root.fifo_extractions = stats.fifo_extractions;
  diagnostics.root.lifo_extractions = stats.lifo_extractions;
  diagnostics.root.tactical_actions = stats.tactical_actions;
  diagnostics.root.proof_classes = stats.proof_classes;
  diagnostics.root.truncations = stats.truncations;
  diagnostics.root.maximum_deque_size = stats.max_deque_size;
  diagnostics.root.deadline_reached = stats.deadline_reached;
  diagnostics.root.proof_truncated = stats.proof_truncated;
  diagnostics.root.exhaustive = diagnostics.generated.exhaustive;
  const RankedAction &best = diagnostics.ranked.front();
  diagnostics.root.initial_best_action =
      diagnostics.generated.actions[best.retained_ordinal].encoded;
  diagnostics.root.initial_best_value = best.action_value;
  return diagnostics;
}

native::SearchConfig native_search_config(const AuditConfig &config,
                                          std::uint32_t time_limit_ms) {
  native::SearchConfig search;
  search.max_actions = config.max_actions;
  search.max_partial_paths = config.max_partial_paths;
  search.max_tree_nodes = config.fixed_work;
  search.root_reply_width = config.root_reply_width;
  search.supported_advance_penalty = config.supported_advance_penalty;
  search.max_expansions = config.max_expansions;
  search.exploration_constant = config.exploration;
  search.first_play_urgency = config.first_play_urgency;
  if (time_limit_ms != 0U) {
    search.absolute_deadline = native::SearchClock::now() +
                               std::chrono::milliseconds(time_limit_ms);
  }
  return search;
}

struct SearchRun {
  native::SearchResult result;
  SearchDecision decision;
};

SearchRun choose_native(const GameState &state, const AuditConfig &config,
                        std::uint32_t time_limit_ms,
                        std::optional<native::QuantizedModel> &game_model,
                        const LoadedRuntime &runtime) {
  const auto started = native::SearchClock::now();
  if (!game_model.has_value()) {
    game_model.emplace(make_runtime_model(runtime));
  }
  native::SearchConfig search_config =
      native_search_config(config, time_limit_ms);
  search_config.model = &*game_model;
  // Include search configuration and immutable-model initialization in the
  // measured decision, just as the standalone bot does after state replay.
  if (time_limit_ms != 0U) {
    search_config.absolute_deadline =
        started + std::chrono::milliseconds(time_limit_ms);
  }
  SearchRun run;
  run.result = native::choose_complete_turn(state, search_config);
  run.decision.elapsed_ms =
      std::chrono::duration<double, std::milli>(native::SearchClock::now() -
                                                started)
          .count();
  run.decision.action = run.result.encoded;
  run.decision.value = run.result.value;
  run.decision.solved = run.result.solved;
  if (run.result.solved_winner.has_value()) {
    run.decision.solved_winner = player_id(*run.result.solved_winner);
  }
  run.decision.root_actions = run.result.root_actions.size();
  for (const native::RootActionStat &action : run.result.root_actions) {
    if (action.exact_reply_refuted) {
      run.decision.exact_reply_refuted_actions.push_back(action.encoded);
    }
    GameState endpoint = state;
    native::apply_encoded_turn(endpoint, action.encoded);
    const bool proven_win =
        native::proven_action_win(action.solved, action.value);
    const bool penalty_eligible = native::supported_advance_penalty_eligible(
        state.ball, endpoint.ball, state.to_move, action.tactical,
        state.used_segments.size(), action.solved);
    const double unpenalized =
        native::final_action_score(action.value, action.visits);
    run.decision.root_action_diagnostics.push_back(
        SearchDecision::RootAction{
            action.encoded,
            action.value,
            action.initial_value,
            action.visits,
            action.selection_visits,
            tactical_class_name(action.tactical),
            action.solved,
            proven_win,
            action.exact_reply_refuted,
            native::mover_normalized_progress(state.ball, state.to_move),
            native::mover_normalized_progress(endpoint.ball, state.to_move),
            penalty_eligible,
            action.penalty_applied,
            unpenalized,
            unpenalized - action.penalty_applied});
  }
  run.decision.stats = run.result.stats;
  return run;
}

int exact_ordinal(const DiagnosticGeneration &diagnostics,
                  std::string_view action) {
  for (std::size_t index = 0; index < diagnostics.generated.actions.size();
       ++index) {
    if (diagnostics.generated.actions[index].encoded == action) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

int boundary_ordinal(const GameState &state,
                     const DiagnosticGeneration &diagnostics,
                     std::string_view action) {
  GameState endpoint = state;
  native::apply_encoded_turn(endpoint, action);
  const native::SearchPosition compact(diagnostics.topology, endpoint);
  for (std::size_t index = 0; index < diagnostics.generated.actions.size();
       ++index) {
    if (diagnostics.generated.actions[index].result.same_compact_state(compact)) {
      return static_cast<int>(index);
    }
  }
  return -1;
}

const RankedAction &ranked_for_ordinal(const DiagnosticGeneration &diagnostics,
                                      std::size_t ordinal) {
  for (const RankedAction &ranked : diagnostics.ranked) {
    if (ranked.retained_ordinal == ordinal) return ranked;
  }
  throw std::logic_error("retained action has no initial rank");
}

ActionDiagnostic inspect_action(
    const GameState &state, std::string_view action,
    const DiagnosticGeneration &diagnostics,
    const native::SearchResult &search_result) {
  ActionDiagnostic result;
  result.action = std::string(action);
  result.exact_retained_ordinal = exact_ordinal(diagnostics, action);
  result.boundary_retained_ordinal =
      boundary_ordinal(state, diagnostics, action);
  if (result.boundary_retained_ordinal < 0) return result;

  const std::size_t ordinal =
      static_cast<std::size_t>(result.boundary_retained_ordinal);
  const native::CompleteTurnAction &retained =
      diagnostics.generated.actions[ordinal];
  result.retained_action = retained.encoded;
  result.tactical_class = tactical_class_name(retained.tactical);
  const RankedAction &ranked = ranked_for_ordinal(diagnostics, ordinal);
  result.initial_neural_value = ranked.neural_value;
  result.initial_action_value = ranked.action_value;
  result.initial_rank = static_cast<int>(ranked.rank);

  const auto copy_final = [&](const native::RootActionStat &root_action) {
    result.final_backed_value = root_action.value;
    result.final_visits = root_action.visits;
    result.final_selection_visits = root_action.selection_visits;
    result.final_tactical_class = tactical_class_name(root_action.tactical);
    result.exact_reply_refuted = root_action.exact_reply_refuted;
    result.final_solved = root_action.solved;
    result.final_proven_win =
        native::proven_action_win(root_action.solved, root_action.value);
    GameState endpoint = state;
    native::apply_encoded_turn(endpoint, root_action.encoded);
    result.start_progress =
        native::mover_normalized_progress(state.ball, state.to_move);
    result.endpoint_progress =
        native::mover_normalized_progress(endpoint.ball, state.to_move);
    result.supported_advance_eligible =
        native::supported_advance_penalty_eligible(
            state.ball, endpoint.ball, state.to_move, root_action.tactical,
            state.used_segments.size(), root_action.solved);
    result.supported_advance_penalty = root_action.penalty_applied;
    result.final_unpenalized_score =
        native::final_action_score(root_action.value, root_action.visits);
    result.final_penalized_score =
        *result.final_unpenalized_score - root_action.penalty_applied;
  };
  for (const native::RootActionStat &root_action : search_result.root_actions) {
    if (root_action.encoded == action) {
      copy_final(root_action);
      return result;
    }
  }
  for (const native::RootActionStat &root_action : search_result.root_actions) {
    if (root_action.encoded == retained.encoded) {
      copy_final(root_action);
      return result;
    }
  }
  return result;
}

std::pair<std::string, std::string> classify_decision(
    const ActionDiagnostic &actual, const ActionDiagnostic &chosen,
    const SearchDecision &search, const RootDiagnostics &root) {
  if (actual.action == chosen.action) {
    return {"match", "the native search reproduced the observed encoding"};
  }
  if (actual.boundary_retained_ordinal >= 0 &&
      actual.boundary_retained_ordinal == chosen.boundary_retained_ordinal) {
    return {"boundary-equivalent",
            "the encodings differ but reach the same complete-turn boundary"};
  }
  if (actual.boundary_retained_ordinal < 0) {
    return {"generator-omission",
            "the deterministic diagnostic generator omitted the observed "
            "boundary"};
  }
  if (chosen.boundary_retained_ordinal < 0 ||
      !chosen.final_backed_value.has_value() ||
      (search.stats.deadline_reached &&
       !actual.final_backed_value.has_value())) {
    return {"operational-failure",
            "deadline or root-set pressure prevented a comparable retained "
            "search action"};
  }
  const int actual_tactical = tactical_order(actual.tactical_class);
  const int chosen_tactical = tactical_order(chosen.tactical_class);
  if ((actual_tactical <= 1 && actual_tactical < chosen_tactical) ||
      (chosen_tactical >= 3 && actual_tactical < chosen_tactical)) {
    return {"tactical-miss",
            "the observed boundary has a stronger exact tactical class than "
            "the chosen boundary"};
  }
  if (chosen.retained_action != root.initial_best_action) {
    return {"bfm-override",
            "BFM selected a boundary other than the initial value leader"};
  }
  return {"initial-evaluator-ordering",
          "the initial native value ordering preferred the chosen boundary"};
}

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) transcript.push_back('/');
  transcript.append(action);
}

std::vector<DecisionAudit> audit_records(
    const std::vector<GameRecord> &records, const AuditConfig &config,
    std::vector<SelectedPrefix> *selected_prefixes = nullptr,
    const LoadedRuntime *runtime_override = nullptr) {
  std::optional<LoadedRuntime> compiled_runtime;
  if (runtime_override == nullptr) compiled_runtime.emplace(default_runtime());
  const LoadedRuntime &runtime =
      runtime_override == nullptr ? *compiled_runtime : *runtime_override;
  std::vector<DecisionAudit> audits;
  for (const GameRecord &record : records) {
    GameState state = make_initial_state(codingame_rules());
    std::optional<native::QuantizedModel> game_model;
    std::string prefix;
    std::size_t own_decision = 0;
    for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
      if (player_id(state.to_move) == record.candidate_player) {
        const std::size_t decision_index = own_decision++;
        bool audit_decision =
            !config.max_own_decisions_per_game.has_value() ||
            decision_index < *config.max_own_decisions_per_game;
        std::string selected_state_id;
        if (selected_prefixes != nullptr) {
          audit_decision = false;
          for (SelectedPrefix &selected : *selected_prefixes) {
            if (selected.game_id != record.game_id ||
                selected.turn_index != turn) continue;
            selected_state_id = state_identity(state);
            if (selected.state_id != selected_state_id) {
              throw std::invalid_argument(
                  "selected-prefix state identity disagrees with replay");
            }
            selected.matched = true;
            audit_decision = true;
            break;
          }
        }
        if (!audit_decision) {
          native::apply_encoded_turn(state, record.actions[turn]);
          append_turn(prefix, record.actions[turn]);
          continue;
        }
        DecisionAudit audit;
        audit.game_id = record.game_id;
        audit.state_id = selected_state_id.empty() ? state_identity(state)
                                                    : selected_state_id;
        audit.transcript_prefix = prefix;
        audit.turn_index = turn;
        audit.own_decision_index = decision_index;
        audit.candidate_player = record.candidate_player;
        audit.winner = record.winner;
        audit.pre_action_used_edges = state.used_segments.size();
        audit.provenance = record.provenance;
        audit.runtime = runtime.identity;
        audit.time_limit_ms = decision_time_limit(config,
                                                  audit.own_decision_index);
        SearchRun search =
            choose_native(state, config, audit.time_limit_ms, game_model,
                          runtime);
        DiagnosticGeneration diagnostics =
            generate_root_diagnostics(state, config, *game_model);
        audit.root = diagnostics.root;
        audit.search = search.decision;
        audit.actual = inspect_action(state, record.actions[turn], diagnostics,
                                      search.result);
        audit.chosen = inspect_action(state, search.result.encoded, diagnostics,
                                      search.result);
        std::tie(audit.classification, audit.classification_reason) =
            classify_decision(audit.actual, audit.chosen, audit.search,
                              audit.root);
        audits.push_back(std::move(audit));
      }
      native::apply_encoded_turn(state, record.actions[turn]);
      append_turn(prefix, record.actions[turn]);
    }
  }
  if (selected_prefixes != nullptr &&
      std::any_of(selected_prefixes->begin(), selected_prefixes->end(),
                  [](const SelectedPrefix &prefix) {
                    return !prefix.matched;
                  })) {
    throw std::invalid_argument(
        "selected-prefix manifest contains an unknown candidate decision");
  }
  return audits;
}

std::string json_quote(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2U);
  result.push_back('"');
  constexpr char hexadecimal[] = "0123456789abcdef";
  for (const unsigned char character : value) {
    switch (character) {
      case '"': result.append("\\\""); break;
      case '\\': result.append("\\\\"); break;
      case '\b': result.append("\\b"); break;
      case '\f': result.append("\\f"); break;
      case '\n': result.append("\\n"); break;
      case '\r': result.append("\\r"); break;
      case '\t': result.append("\\t"); break;
      default:
        if (character < 0x20U) {
          result.append("\\u00");
          result.push_back(hexadecimal[character >> 4U]);
          result.push_back(hexadecimal[character & 0x0fU]);
        } else {
          result.push_back(static_cast<char>(character));
        }
    }
  }
  result.push_back('"');
  return result;
}

std::string provenance_json(const ProvenanceMetadata &provenance) {
  std::string result{"{"};
  for (std::size_t index = 0; index < provenance.size(); ++index) {
    if (index != 0U) result.push_back(',');
    result.append(json_quote(provenance[index].first));
    result.push_back(':');
    result.append(json_quote(provenance[index].second));
  }
  result.push_back('}');
  return result;
}

std::string color_name(int player) {
  return player == 0 ? "player_one" : "player_two";
}

std::string result_name(const DecisionAudit &audit) {
  return audit.candidate_player == audit.winner ? "win" : "loss";
}

template <typename Value>
void write_json_optional(std::ostream &output,
                         const std::optional<Value> &value) {
  if (value.has_value()) output << *value;
  else output << "null";
}

template <typename Value>
void write_tsv_optional(std::ostream &output,
                        const std::optional<Value> &value) {
  if (value.has_value()) output << *value;
}

void write_json_action(std::ostream &output, std::string_view prefix,
                       const ActionDiagnostic &action) {
  output << json_quote(std::string(prefix) + "_action") << ':'
         << json_quote(action.action) << ','
         << json_quote(std::string(prefix) + "_exact_retained_ordinal") << ':'
         << action.exact_retained_ordinal << ','
         << json_quote(std::string(prefix) + "_boundary_retained_ordinal")
         << ':' << action.boundary_retained_ordinal << ','
         << json_quote(std::string(prefix) + "_retained_action") << ':'
         << json_quote(action.retained_action) << ','
         << json_quote(std::string(prefix) + "_tactical_class") << ':'
         << json_quote(action.tactical_class) << ','
         << json_quote(std::string(prefix) + "_initial_neural_value") << ':';
  write_json_optional(output, action.initial_neural_value);
  output << ',' << json_quote(std::string(prefix) + "_initial_action_value")
         << ':';
  write_json_optional(output, action.initial_action_value);
  output << ',' << json_quote(std::string(prefix) + "_initial_rank") << ':'
         << action.initial_rank << ','
         << json_quote(std::string(prefix) + "_final_backed_value") << ':';
  write_json_optional(output, action.final_backed_value);
  output << ',' << json_quote(std::string(prefix) + "_final_visits") << ':';
  write_json_optional(output, action.final_visits);
  output << ','
         << json_quote(std::string(prefix) + "_final_selection_visits") << ':';
  write_json_optional(output, action.final_selection_visits);
  output << ',' << json_quote(std::string(prefix) + "_final_tactical_class")
         << ':' << json_quote(action.final_tactical_class) << ','
         << json_quote(std::string(prefix) + "_exact_reply_refuted") << ':'
         << (action.exact_reply_refuted ? "true" : "false") << ','
         << json_quote(std::string(prefix) + "_final_solved") << ':'
         << (action.final_solved ? "true" : "false") << ','
         << json_quote(std::string(prefix) + "_final_proven_win") << ':'
         << (action.final_proven_win ? "true" : "false") << ','
         << json_quote(std::string(prefix) + "_start_progress") << ':';
  write_json_optional(output, action.start_progress);
  output << ',' << json_quote(std::string(prefix) + "_endpoint_progress")
         << ':';
  write_json_optional(output, action.endpoint_progress);
  output << ','
         << json_quote(std::string(prefix) +
                       "_supported_advance_eligible")
         << ':' << (action.supported_advance_eligible ? "true" : "false")
         << ','
         << json_quote(std::string(prefix) +
                       "_supported_advance_penalty")
         << ':' << action.supported_advance_penalty << ','
         << json_quote(std::string(prefix) + "_final_unpenalized_score")
         << ':';
  write_json_optional(output, action.final_unpenalized_score);
  output << ','
         << json_quote(std::string(prefix) + "_final_penalized_score")
         << ':';
  write_json_optional(output, action.final_penalized_score);
}

std::string action_list_json(const std::vector<std::string> &actions) {
  std::string result{"["};
  for (std::size_t index = 0; index < actions.size(); ++index) {
    if (index != 0) result.push_back(',');
    result.append(json_quote(actions[index]));
  }
  result.push_back(']');
  return result;
}

std::string action_list_text(const std::vector<std::string> &actions) {
  std::string result;
  for (const std::string &action : actions) {
    if (!result.empty()) result.push_back(',');
    result.append(action);
  }
  return result;
}

void write_json_root_actions(
    std::ostream &output,
    const std::vector<SearchDecision::RootAction> &actions) {
  output << std::setprecision(std::numeric_limits<double>::max_digits10) << '[';
  for (std::size_t index = 0; index < actions.size(); ++index) {
    if (index != 0) output << ',';
    const SearchDecision::RootAction &action = actions[index];
    output << '{' << "\"encoded\":" << json_quote(action.encoded) << ','
           << "\"value\":" << action.value << ',' << "\"initial_value\":"
           << action.initial_value << ',' << "\"visits\":"
           << action.visits << ',' << "\"selection_visits\":"
           << action.selection_visits << ',' << "\"tactical_class\":"
           << json_quote(action.tactical_class) << ',' << "\"solved\":"
           << (action.solved ? "true" : "false") << ','
           << "\"proven_win\":"
           << (action.proven_win ? "true" : "false") << ','
           << "\"exact_reply_refuted\":"
           << (action.exact_reply_refuted ? "true" : "false") << ','
           << "\"start_progress\":" << action.start_progress << ','
           << "\"endpoint_progress\":" << action.endpoint_progress << ','
           << "\"supported_advance_eligible\":"
           << (action.supported_advance_eligible ? "true" : "false") << ','
           << "\"supported_advance_penalty\":"
           << action.supported_advance_penalty << ','
           << "\"unpenalized_final_score\":"
           << action.unpenalized_final_score << ','
           << "\"penalized_final_score\":" << action.penalized_final_score
           << '}';
  }
  output << ']';
}

void write_json_line(std::ostream &output, const DecisionAudit &audit,
                     const AuditConfig &config) {
  output << '{'
         << "\"schema_version\":" << json_quote(kSchemaVersion) << ','
         << "\"input_provenance\":" << provenance_json(audit.provenance)
         << ',' << "\"game_id\":" << json_quote(audit.game_id) << ','
         << "\"state_id\":" << json_quote(audit.state_id) << ','
         << "\"transcript_prefix\":"
         << json_quote(audit.transcript_prefix) << ','
         << "\"turn_index\":" << audit.turn_index << ','
         << "\"own_decision_index\":" << audit.own_decision_index << ','
         << "\"candidate_player\":" << audit.candidate_player << ','
         << "\"color\":" << json_quote(color_name(audit.candidate_player))
         << ',' << "\"winner\":" << audit.winner << ','
         << "\"result\":" << json_quote(result_name(audit)) << ','
         << "\"classification\":" << json_quote(audit.classification) << ','
         << "\"classification_reason\":"
         << json_quote(audit.classification_reason) << ','
         << "\"audit_mode\":"
         << json_quote(config.mode == AuditMode::Clock ? "clock" :
                                                      "fixed-work")
         << ',' << "\"runtime_sha256\":"
         << json_quote(audit.runtime.runtime_sha256) << ','
         << "\"model_sha256\":"
         << json_quote(audit.runtime.model_sha256) << ','
         << "\"packed_weights_sha256\":"
         << json_quote(audit.runtime.packed_sha256) << ','
         << "\"fixed_work_limit\":" << config.fixed_work << ','
         << "\"max_actions\":" << config.max_actions << ','
         << "\"max_partial_paths\":" << config.max_partial_paths << ','
         << "\"max_expansions\":" << config.max_expansions << ','
         << "\"root_reply_width\":" << config.root_reply_width << ','
         << "\"supported_advance_penalty\":" << std::setprecision(9)
         << config.supported_advance_penalty << ','
         << "\"supported_advance_sparse_edge_limit\":"
         << native::kSupportedAdvanceSparseEdgeLimit << ','
         << "\"supported_advance_start_progress\":"
         << native::kSupportedAdvanceStartProgress << ','
         << "\"supported_advance_endpoint_progress_threshold\":"
         << native::kSupportedAdvanceEndpointProgressThreshold << ','
         << "\"maximum_supported_advance_penalty\":"
         << native::kMaximumSupportedAdvancePenalty << ','
         << "\"pre_action_used_edges\":" << audit.pre_action_used_edges << ','
         << "\"exploration\":" << std::setprecision(9)
         << config.exploration << ',' << "\"first_play_urgency\":"
         << config.first_play_urgency << ','
         << "\"first_time_limit_ms\":" << config.first_time_ms << ','
         << "\"later_time_limit_ms\":" << config.later_time_ms << ','
         << "\"max_own_decisions_per_game\":";
  write_json_optional(output, config.max_own_decisions_per_game);
  output << ','
         << "\"time_limit_ms\":" << audit.time_limit_ms << ',';
  write_json_action(output, "actual", audit.actual);
  output << ',';
  write_json_action(output, "chosen", audit.chosen);
  const native::SearchStats &stats = audit.search.stats;
  output << ',' << "\"chosen_value\":" << audit.search.value << ','
         << "\"search_solved\":"
         << (audit.search.solved ? "true" : "false") << ','
         << "\"search_solved_winner\":";
  write_json_optional(output, audit.search.solved_winner);
  output << ',' << "\"search_elapsed_ms\":" << std::setprecision(9)
         << audit.search.elapsed_ms << ','
         << "\"search_root_actions\":" << audit.search.root_actions << ','
         << "\"search_tree_nodes\":" << stats.tree_nodes << ','
         << "\"search_expansions\":" << stats.expansions << ','
         << "\"search_generated_children\":" << stats.generated_children
         << ',' << "\"search_child_evaluations\":"
         << stats.child_evaluations << ','
         << "\"search_tactical_child_values\":"
         << stats.tactical_child_values << ','
         << "\"search_generator_partial_paths\":"
         << stats.generator_partial_paths << ','
         << "\"search_tactical_proof_paths\":" << stats.proof_paths << ','
         << "\"search_completed_actions\":" << stats.completed_actions
         << ',' << "\"search_duplicate_boundaries\":"
         << stats.duplicate_boundaries << ','
         << "\"search_fifo_extractions\":"
         << stats.generator_fifo_extractions << ','
         << "\"search_lifo_extractions\":"
         << stats.generator_lifo_extractions << ','
         << "\"search_tactical_actions\":" << stats.tactical_actions << ','
         << "\"search_tactical_classes_found\":" << stats.proof_classes
         << ',' << "\"search_tactical_proof_truncations\":"
         << stats.proof_truncations << ','
         << "\"search_generator_truncations\":"
         << stats.generator_truncations << ','
         << "\"search_root_reply_probe_candidates\":"
         << stats.root_reply_probe_candidates << ','
         << "\"search_root_reply_probe_attempts\":"
         << stats.root_reply_probe_attempts << ','
         << "\"search_root_reply_probe_expansions\":"
         << stats.root_reply_probe_expansions << ','
         << "\"search_root_reply_probe_refutations\":"
         << stats.root_reply_probe_refutations << ','
         << "\"search_root_reply_probe_generator_truncations\":"
         << stats.root_reply_probe_generator_truncations << ','
         << "\"search_root_reply_probe_budget_truncations\":"
         << stats.root_reply_probe_budget_truncations << ','
         << "\"search_exact_reply_refuted_actions\":"
         << action_list_json(audit.search.exact_reply_refuted_actions) << ','
         << "\"search_root_action_diagnostics\":";
  write_json_root_actions(output, audit.search.root_action_diagnostics);
  output << ','
         << "\"search_deadline_reached\":"
         << (stats.deadline_reached ? "true" : "false") << ','
         << "\"search_tree_cap_reached\":"
         << (stats.tree_cap_reached ? "true" : "false") << ','
         << "\"search_expansion_cap_reached\":"
         << (stats.expansion_cap_reached ? "true" : "false") << ','
         << "\"diagnostic_root_actions\":" << audit.root.retained_actions
         << ',' << "\"diagnostic_root_partial_paths\":"
         << audit.root.explored_partial_paths << ','
         << "\"diagnostic_root_tactical_proof_paths\":"
         << audit.root.proof_paths << ','
         << "\"diagnostic_root_completed_actions\":"
         << audit.root.completed_actions << ','
         << "\"diagnostic_root_duplicate_boundaries\":"
         << audit.root.duplicate_boundaries << ','
         << "\"diagnostic_root_fifo_extractions\":"
         << audit.root.fifo_extractions << ','
         << "\"diagnostic_root_lifo_extractions\":"
         << audit.root.lifo_extractions << ','
         << "\"diagnostic_root_tactical_actions\":"
         << audit.root.tactical_actions << ','
         << "\"diagnostic_root_tactical_classes_found\":"
         << static_cast<unsigned>(audit.root.proof_classes) << ','
         << "\"diagnostic_root_tactical_proof_truncated\":"
         << (audit.root.proof_truncated ? "true" : "false") << ','
         << "\"diagnostic_root_truncations\":" << audit.root.truncations
         << ',' << "\"diagnostic_root_maximum_deque_size\":"
         << audit.root.maximum_deque_size << ','
         << "\"diagnostic_root_deadline_reached\":"
         << (audit.root.deadline_reached ? "true" : "false") << ','
         << "\"diagnostic_root_exhaustive\":"
         << (audit.root.exhaustive ? "true" : "false") << ','
         << "\"initial_best_action\":"
         << json_quote(audit.root.initial_best_action) << ','
         << "\"initial_best_value\":" << audit.root.initial_best_value
         << "}\n";
}

inline constexpr std::string_view kTsvHeader =
    "schema_version\tinput_provenance_json\tgame_id\tstate_id\t"
    "transcript_prefix\tturn_index\town_decision_index\tcandidate_player\t"
    "color\twinner\tresult\tclassification\tclassification_reason\t"
    "audit_mode\tmodel_sha256\tpacked_weights_sha256\tfixed_work_limit\t"
    "max_actions\tmax_partial_paths\tmax_expansions\texploration\t"
    "first_play_urgency\tfirst_time_limit_ms\tlater_time_limit_ms\t"
    "max_own_decisions_per_game\ttime_limit_ms\tactual_action\t"
    "actual_exact_retained_ordinal\t"
    "actual_boundary_retained_ordinal\tactual_retained_action\t"
    "actual_tactical_class\tactual_initial_neural_value\t"
    "actual_initial_action_value\tactual_initial_rank\t"
    "actual_final_backed_value\tactual_final_visits\t"
    "actual_final_selection_visits\tactual_final_tactical_class\t"
    "chosen_action\t"
    "chosen_exact_retained_ordinal\tchosen_boundary_retained_ordinal\t"
    "chosen_retained_action\tchosen_tactical_class\t"
    "chosen_initial_neural_value\tchosen_initial_action_value\t"
    "chosen_initial_rank\tchosen_final_backed_value\tchosen_final_visits\t"
    "chosen_final_selection_visits\tchosen_final_tactical_class\t"
    "chosen_value\tsearch_solved\t"
    "search_solved_winner\tsearch_elapsed_ms\tsearch_root_actions\t"
    "search_tree_nodes\tsearch_expansions\tsearch_generated_children\t"
    "search_child_evaluations\tsearch_tactical_child_values\t"
    "search_generator_partial_paths\tsearch_tactical_proof_paths\t"
    "search_completed_actions\t"
    "search_duplicate_boundaries\tsearch_fifo_extractions\t"
    "search_lifo_extractions\tsearch_tactical_actions\t"
    "search_tactical_classes_found\tsearch_tactical_proof_truncations\t"
    "search_generator_truncations\tsearch_deadline_reached\t"
    "search_tree_cap_reached\tsearch_expansion_cap_reached\t"
    "diagnostic_root_actions\tdiagnostic_root_partial_paths\t"
    "diagnostic_root_tactical_proof_paths\t"
    "diagnostic_root_completed_actions\tdiagnostic_root_duplicate_boundaries\t"
    "diagnostic_root_fifo_extractions\tdiagnostic_root_lifo_extractions\t"
    "diagnostic_root_tactical_actions\t"
    "diagnostic_root_tactical_classes_found\t"
    "diagnostic_root_tactical_proof_truncated\t"
    "diagnostic_root_truncations\t"
    "diagnostic_root_maximum_deque_size\tdiagnostic_root_deadline_reached\t"
    "diagnostic_root_exhaustive\tinitial_best_action\tinitial_best_value\t"
    "root_reply_width\tactual_exact_reply_refuted\t"
    "chosen_exact_reply_refuted\tsearch_root_reply_probe_candidates\t"
    "search_root_reply_probe_attempts\tsearch_root_reply_probe_expansions\t"
    "search_root_reply_probe_refutations\t"
    "search_root_reply_probe_generator_truncations\t"
    "search_root_reply_probe_budget_truncations\t"
    "search_exact_reply_refuted_actions\truntime_sha256\t"
    "supported_advance_penalty\tsupported_advance_sparse_edge_limit\t"
    "supported_advance_start_progress\t"
    "supported_advance_endpoint_progress_threshold\t"
    "maximum_supported_advance_penalty\tpre_action_used_edges\t"
    "actual_final_solved\tactual_final_proven_win\tactual_start_progress\t"
    "actual_endpoint_progress\tactual_supported_advance_eligible\t"
    "actual_supported_advance_penalty\tactual_final_unpenalized_score\t"
    "actual_final_penalized_score\tchosen_final_solved\t"
    "chosen_final_proven_win\tchosen_start_progress\t"
    "chosen_endpoint_progress\tchosen_supported_advance_eligible\t"
    "chosen_supported_advance_penalty\tchosen_final_unpenalized_score\t"
    "chosen_final_penalized_score\tsearch_root_action_diagnostics_json";

void write_tsv_action(std::ostream &output, const ActionDiagnostic &action) {
  output << action.action << '\t' << action.exact_retained_ordinal << '\t'
         << action.boundary_retained_ordinal << '\t' << action.retained_action
         << '\t' << action.tactical_class << '\t';
  write_tsv_optional(output, action.initial_neural_value);
  output << '\t';
  write_tsv_optional(output, action.initial_action_value);
  output << '\t' << action.initial_rank << '\t';
  write_tsv_optional(output, action.final_backed_value);
  output << '\t';
  write_tsv_optional(output, action.final_visits);
  output << '\t';
  write_tsv_optional(output, action.final_selection_visits);
  output << '\t' << action.final_tactical_class;
}

void write_tsv_supported_advance(std::ostream &output,
                                 const ActionDiagnostic &action) {
  output << action.final_solved << '\t' << action.final_proven_win << '\t';
  write_tsv_optional(output, action.start_progress);
  output << '\t';
  write_tsv_optional(output, action.endpoint_progress);
  output << '\t' << action.supported_advance_eligible << '\t'
         << action.supported_advance_penalty << '\t';
  write_tsv_optional(output, action.final_unpenalized_score);
  output << '\t';
  write_tsv_optional(output, action.final_penalized_score);
}

void write_tsv_line(std::ostream &output, const DecisionAudit &audit,
                    const AuditConfig &config) {
  output << kSchemaVersion << '\t' << provenance_json(audit.provenance) << '\t'
         << audit.game_id << '\t' << audit.state_id << '\t'
         << audit.transcript_prefix << '\t' << audit.turn_index << '\t'
         << audit.own_decision_index << '\t' << audit.candidate_player << '\t'
         << color_name(audit.candidate_player) << '\t' << audit.winner << '\t'
         << result_name(audit) << '\t' << audit.classification << '\t'
         << audit.classification_reason << '\t'
         << (config.mode == AuditMode::Clock ? "clock" : "fixed-work") << '\t'
         << audit.runtime.model_sha256 << '\t'
         << audit.runtime.packed_sha256 << '\t'
         << config.fixed_work << '\t' << config.max_actions << '\t'
         << config.max_partial_paths << '\t' << config.max_expansions << '\t'
         << std::setprecision(9) << config.exploration << '\t'
         << config.first_play_urgency << '\t' << config.first_time_ms << '\t'
         << config.later_time_ms << '\t';
  write_tsv_optional(output, config.max_own_decisions_per_game);
  output << '\t' << audit.time_limit_ms << '\t';
  write_tsv_action(output, audit.actual);
  output << '\t';
  write_tsv_action(output, audit.chosen);
  const native::SearchStats &stats = audit.search.stats;
  output << '\t' << audit.search.value << '\t' << audit.search.solved << '\t';
  write_tsv_optional(output, audit.search.solved_winner);
  output << '\t' << std::setprecision(9) << audit.search.elapsed_ms << '\t'
         << audit.search.root_actions << '\t' << stats.tree_nodes << '\t'
         << stats.expansions << '\t' << stats.generated_children << '\t'
         << stats.child_evaluations << '\t' << stats.tactical_child_values
         << '\t' << stats.generator_partial_paths << '\t' << stats.proof_paths
         << '\t' << stats.completed_actions << '\t' << stats.duplicate_boundaries
         << '\t' << stats.generator_fifo_extractions << '\t'
         << stats.generator_lifo_extractions << '\t' << stats.tactical_actions
         << '\t' << stats.proof_classes << '\t' << stats.proof_truncations
         << '\t' << stats.generator_truncations << '\t'
         << stats.deadline_reached << '\t' << stats.tree_cap_reached << '\t'
         << stats.expansion_cap_reached << '\t' << audit.root.retained_actions
         << '\t' << audit.root.explored_partial_paths << '\t'
         << audit.root.proof_paths << '\t'
         << audit.root.completed_actions << '\t'
         << audit.root.duplicate_boundaries << '\t'
         << audit.root.fifo_extractions << '\t' << audit.root.lifo_extractions
         << '\t' << audit.root.tactical_actions << '\t'
         << static_cast<unsigned>(audit.root.proof_classes) << '\t'
         << audit.root.proof_truncated << '\t' << audit.root.truncations << '\t'
         << audit.root.maximum_deque_size
         << '\t' << audit.root.deadline_reached << '\t'
         << audit.root.exhaustive << '\t' << audit.root.initial_best_action
         << '\t' << audit.root.initial_best_value << '\t'
         << config.root_reply_width << '\t'
         << audit.actual.exact_reply_refuted << '\t'
         << audit.chosen.exact_reply_refuted << '\t'
         << stats.root_reply_probe_candidates << '\t'
         << stats.root_reply_probe_attempts << '\t'
         << stats.root_reply_probe_expansions << '\t'
         << stats.root_reply_probe_refutations << '\t'
         << stats.root_reply_probe_generator_truncations << '\t'
         << stats.root_reply_probe_budget_truncations << '\t'
         << action_list_text(audit.search.exact_reply_refuted_actions) << '\t'
         << audit.runtime.runtime_sha256 << '\t'
         << config.supported_advance_penalty << '\t'
         << native::kSupportedAdvanceSparseEdgeLimit << '\t'
         << native::kSupportedAdvanceStartProgress << '\t'
         << native::kSupportedAdvanceEndpointProgressThreshold << '\t'
         << native::kMaximumSupportedAdvancePenalty << '\t'
         << audit.pre_action_used_edges << '\t';
  write_tsv_supported_advance(output, audit.actual);
  output << '\t';
  write_tsv_supported_advance(output, audit.chosen);
  output << '\t';
  write_json_root_actions(output, audit.search.root_action_diagnostics);
  output << '\n';
}

void write_audits(std::ostream &output,
                  const std::vector<DecisionAudit> &audits,
                  const AuditConfig &config) {
  if (config.output_format == OutputFormat::Tsv) output << kTsvHeader << '\n';
  for (const DecisionAudit &audit : audits) {
    if (config.output_format == OutputFormat::JsonLines) {
      write_json_line(output, audit, config);
    } else {
      write_tsv_line(output, audit, config);
    }
  }
}

void validate_config(const AuditConfig &config) {
  if (config.fixed_work < 2 ||
      config.fixed_work > native::kMaximumTreeNodes ||
      config.max_actions == 0 || config.max_actions > native::kMaximumActions ||
      config.max_partial_paths == 0 ||
      config.max_partial_paths > native::kMaximumPartialPaths ||
      config.max_expansions == 0 ||
      config.max_expansions > native::kMaximumExpansions ||
      config.root_reply_width > config.max_actions ||
      !std::isfinite(config.supported_advance_penalty) ||
      config.supported_advance_penalty < 0.0 ||
      config.supported_advance_penalty >
          native::kMaximumSupportedAdvancePenalty ||
      !std::isfinite(config.exploration) || config.exploration < 0.0 ||
      !std::isfinite(config.first_play_urgency)) {
    throw std::invalid_argument("invalid native replay-audit configuration");
  }
  if (config.max_own_decisions_per_game.has_value() &&
      (*config.max_own_decisions_per_game == 0 ||
       *config.max_own_decisions_per_game > kMaximumTurns)) {
    throw std::invalid_argument(
        "maximum own decisions per game must be in [1,1024]");
  }
  if (config.mode == AuditMode::FixedWork &&
      (config.first_time_ms != 0 || config.later_time_ms != 0)) {
    throw std::invalid_argument("fixed-work mode cannot have a clock");
  }
  if (config.mode == AuditMode::Clock &&
      (config.first_time_ms == 0 || config.later_time_ms == 0)) {
    throw std::invalid_argument("clock mode requires positive first/later limits");
  }
}

std::string_view require_value(int &index, int argc, char **argv,
                               std::string_view option) {
  if (++index >= argc) {
    throw std::invalid_argument("missing value for " + std::string(option));
  }
  return argv[index];
}

AuditConfig parse_arguments(int argc, char **argv) {
  AuditConfig config;
  bool custom_clock = false;
  bool codingame_clocks = false;
  bool fixed_work_option = false;
  bool tree_nodes_option = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--input") {
      config.input_path = std::string(require_value(index, argc, argv, option));
    } else if (option == "--checkpoint") {
      config.checkpoint_path =
          std::string(require_value(index, argc, argv, option));
    } else if (option == "--selected-prefixes") {
      config.selected_prefixes_path =
          std::string(require_value(index, argc, argv, option));
    } else if (option == "--format") {
      const std::string_view format = require_value(index, argc, argv, option);
      if (format == "jsonl") config.output_format = OutputFormat::JsonLines;
      else if (format == "tsv") config.output_format = OutputFormat::Tsv;
      else throw std::invalid_argument("--format must be jsonl or tsv");
    } else if (option == "--fixed-work") {
      config.fixed_work = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "fixed work");
      fixed_work_option = true;
    } else if (option == "--tree-nodes") {
      config.fixed_work = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "tree nodes");
      tree_nodes_option = true;
    } else if (option == "--max-own-decisions-per-game") {
      config.max_own_decisions_per_game = parse_integer<std::size_t>(
          require_value(index, argc, argv, option),
          "maximum own decisions per game");
    } else if (option == "--max-actions") {
      config.max_actions = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "maximum actions");
    } else if (option == "--max-partial-paths") {
      config.max_partial_paths = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "maximum partial paths");
    } else if (option == "--max-expansions") {
      config.max_expansions = parse_integer<std::uint64_t>(
          require_value(index, argc, argv, option), "maximum expansions");
    } else if (option == "--root-reply-width") {
      config.root_reply_width = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "root reply width");
    } else if (option == "--supported-advance-penalty") {
      config.supported_advance_penalty = parse_double(
          require_value(index, argc, argv, option),
          "supported advance penalty");
    } else if (option == "--exploration") {
      config.exploration = parse_double(
          require_value(index, argc, argv, option), "exploration");
    } else if (option == "--fpu") {
      config.first_play_urgency = parse_double(
          require_value(index, argc, argv, option), "FPU");
    } else if (option == "--first-ms") {
      config.first_time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "first clock");
      custom_clock = true;
    } else if (option == "--later-ms") {
      config.later_time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "later clock");
      custom_clock = true;
    } else if (option == "--codingame-clocks") {
      codingame_clocks = true;
    } else if (option == "--help") {
      throw std::invalid_argument("help");
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (codingame_clocks && custom_clock) {
    throw std::invalid_argument(
        "--codingame-clocks cannot be combined with custom clocks");
  }
  if (fixed_work_option && (codingame_clocks || custom_clock)) {
    throw std::invalid_argument(
        "--fixed-work cannot be combined with clock mode; use --tree-nodes");
  }
  if (codingame_clocks) {
    config.mode = AuditMode::Clock;
    if (!tree_nodes_option) {
      config.fixed_work = native::kProductionTreeNodes;
    }
    config.max_actions = native::kMaximumActions;
    config.max_partial_paths = native::kProductionPartialPaths;
    config.max_expansions = native::kMaximumExpansions;
    config.first_time_ms = native::kFirstSearchTimeMs;
    config.later_time_ms = native::kLaterSearchTimeMs;
  } else if (custom_clock) {
    config.mode = AuditMode::Clock;
  }
  validate_config(config);
  return config;
}

void write_usage(std::ostream &output, std::string_view executable) {
  output << "usage: " << executable << " [options] < games.tsv\n"
         << "input header: " << kInputHeader << "\n"
         << "options:\n"
         << "  --input PATH\n"
         << "  --checkpoint PATH      strict exported runtime checkpoint\n"
         << "  --selected-prefixes PATH\n"
         << "                         game_id/turn_index/state_id TSV filter\n"
         << "  --format jsonl|tsv\n"
         << "  --fixed-work N        deterministic tree-node work (default 30000)\n"
         << "  --tree-nodes N        tree cap, including with clock mode\n"
         << "  --max-own-decisions-per-game N\n"
         << "                         audit only the first N candidate decisions;\n"
         << "                         full transcript validation remains mandatory\n"
         << "  --max-actions N\n"
         << "  --max-partial-paths N\n"
         << "  --max-expansions N\n"
         << "  --root-reply-width K\n"
         << "  --supported-advance-penalty X\n"
         << "  --exploration X\n"
         << "  --fpu X\n"
         << "  --first-ms N --later-ms N\n"
         << "  --codingame-clocks    production 800/155 ms configuration\n";
}

int run(int argc, char **argv, std::istream &standard_input,
        std::ostream &output, std::ostream &error) {
  try {
    const AuditConfig config = parse_arguments(argc, argv);
    const LoadedRuntime runtime = load_runtime(config.checkpoint_path);
    std::ifstream file;
    std::istream *input = &standard_input;
    if (config.input_path.has_value()) {
      file.open(*config.input_path);
      if (!file) {
        throw std::invalid_argument("cannot open input: " +
                                    *config.input_path);
      }
      input = &file;
    }
    const std::vector<GameRecord> records = read_records(*input);
    validate_records(records);
    std::ifstream selected_file;
    std::vector<SelectedPrefix> selected;
    std::vector<SelectedPrefix> *selected_pointer = nullptr;
    if (config.selected_prefixes_path.has_value()) {
      selected_file.open(*config.selected_prefixes_path);
      if (!selected_file) {
        throw std::invalid_argument("cannot open selected-prefix manifest: " +
                                    *config.selected_prefixes_path);
      }
      selected = read_selected_prefixes(selected_file);
      selected_pointer = &selected;
    }
    const std::vector<DecisionAudit> audits =
        audit_records(records, config, selected_pointer, &runtime);
    write_audits(output, audits, config);
    return 0;
  } catch (const std::invalid_argument &error_value) {
    if (std::string_view(error_value.what()) == "help") {
      write_usage(output, argc > 0 ? argv[0] : "replay_decision_auditor");
      return 0;
    }
    error << "error: " << error_value.what() << '\n';
    return 2;
  } catch (const std::exception &error_value) {
    error << "error: " << error_value.what() << '\n';
    return 1;
  }
}

}  // namespace papersoccer::jacek_native_replay_audit

#ifndef PAPER_SOCCER_JACEK_NATIVE_REPLAY_AUDITOR_NO_MAIN
int main(int argc, char **argv) {
  return papersoccer::jacek_native_replay_audit::run(
      argc, argv, std::cin, std::cout, std::cerr);
}
#endif
