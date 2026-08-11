#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_replay_audit_reference
#include "../rank_4/bot.cpp"
#undef turn_action_v2

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank4_replay_audit_candidate
#include "bot.cpp"
#undef turn_action_v2

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace ps = papersoccer;
namespace audit_candidate = papersoccer::rank4_replay_audit_candidate;
namespace audit_reference = papersoccer::rank4_replay_audit_reference;

namespace papersoccer::fullturn_replay_audit {

constexpr std::string_view kSchemaVersion = "fullturn-decision-audit-v2";
constexpr std::string_view kInputHeader =
    "game_id\tcandidate_player\twinner\tturns";
constexpr std::size_t kMaximumGames = 4'096;
constexpr std::size_t kMaximumGameIdBytes = 128;
constexpr std::size_t kMaximumTranscriptBytes = 65'536;
constexpr std::size_t kMaximumTurns = 1'024;
constexpr std::size_t kMaximumActionBytes = 1'024;
constexpr std::size_t kMaximumProvenanceEntries = 64;
constexpr std::size_t kMaximumProvenanceKeyBytes = 64;
constexpr std::size_t kMaximumProvenanceValueBytes = 512;

enum class OutputFormat { JsonLines, Tsv };

using ProvenanceMetadata = std::vector<std::pair<std::string, std::string>>;

struct AuditConfig {
  std::uint64_t candidate_work{30'000};
  std::size_t candidate_tree_nodes{audit_candidate::kMaximumTreeNodes};
  std::size_t candidate_max_actions{audit_candidate::kMaximumActions};
  std::size_t candidate_nonroot_actions{};
  std::uint64_t candidate_max_partial_paths{
      audit_candidate::kMaximumPartialPaths};
  std::uint32_t candidate_first_time_ms{};
  std::uint32_t candidate_later_time_ms{};
  double candidate_exploration{audit_candidate::kExplorationConstant};
  double candidate_fpu{audit_candidate::kFirstPlayUrgency};
  double candidate_final_visit_weight{};
  bool candidate_root_only{};
  std::uint64_t reference_nodes{30'000};
  std::uint32_t reference_depth{audit_reference::kMaximumTurnDepth};
  std::uint32_t reference_first_time_ms{};
  std::uint32_t reference_later_time_ms{};
  int replay_value_blend_percent{15};
  int teacher_residual_weight_percent{100};
  OutputFormat output_format{OutputFormat::JsonLines};
  std::optional<std::string> input_path{};
  bool codingame_clocks{};
};

struct GameRecord {
  std::string game_id;
  int candidate_player{};
  int winner{};
  std::vector<std::string> actions;
  ProvenanceMetadata provenance;
};

struct ReplayCorrection {
  bool lookup_found{};
  bool valid{};
  std::string lookup_action;
  std::string action;
};

struct CandidateDecision {
  std::string action;
  std::uint64_t work{};
  std::uint64_t tree_nodes{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t generator_partial_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t generator_duplicates{};
  std::uint64_t tactical_actions{};
  std::uint64_t generator_truncations{};
  int root_score{};
  bool budget_exhausted{};
  bool deadline_reached{};
  bool node_cap_reached{};
  double elapsed_ms{};
};

struct ReferenceDecision {
  std::string action;
  std::uint64_t nodes{};
  std::uint32_t completed_turn_depth{};
  std::uint32_t attempted_turn_depth{};
  int root_score{};
  bool budget_exhausted{};
  double elapsed_ms{};
};

struct RetainedActionMatch {
  int exact_ordinal{-1};
  int boundary_ordinal{-1};
  std::string tactical_class{"not-retained"};
};

struct RootCoverage {
  std::uint64_t actions{};
  std::uint64_t partial_paths{};
  std::uint64_t completed_actions{};
  std::uint64_t duplicates{};
  std::uint64_t truncations{};
  bool exhaustive{};
  RetainedActionMatch actual;
  RetainedActionMatch candidate;
  RetainedActionMatch reference;
};

struct DecisionAudit {
  std::string game_id;
  std::string state_id;
  std::string transcript_prefix;
  std::size_t turn_index{};
  std::size_t own_decision_index{};
  int candidate_player{};
  int winner{};
  std::string actual_action;
  ProvenanceMetadata provenance;
  ReplayCorrection replay_correction;
  std::string reconstructed_deployed_action;
  std::uint32_t candidate_time_limit_ms{};
  std::uint32_t reference_time_limit_ms{};
  CandidateDecision candidate;
  ReferenceDecision reference;
  RootCoverage root_coverage;
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
    if (end == std::string_view::npos) {
      return result;
    }
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

bool safe_game_id(std::string_view value) noexcept {
  if (value.empty() || value.size() > kMaximumGameIdBytes) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9') || character == '-' ||
           character == '_' || character == '.' || character == ':';
  });
}

bool safe_provenance_key(std::string_view value) noexcept {
  if (value.empty() || value.size() > kMaximumProvenanceKeyBytes) {
    return false;
  }
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
  while (!line.empty() && line.front() == ' ') {
    line.remove_prefix(1U);
  }
  const std::size_t separator = line.find('=');
  if (separator == std::string_view::npos) {
    return;
  }
  const std::string_view key = line.substr(0, separator);
  const std::string_view value = line.substr(separator + 1U);
  if (!safe_provenance_key(key) || value.empty() ||
      value.size() > kMaximumProvenanceValueBytes ||
      !std::all_of(value.begin(), value.end(), [](char character) {
        return character >= 0x20 && character <= 0x7e;
      })) {
    throw std::invalid_argument(
        "line " + std::to_string(line_number) +
        " has invalid provenance metadata; expected # safe_key=printable_value");
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
                                " requires candidate_player and winner in {0,1}");
  }
  if (fields[3].empty() || fields[3].size() > kMaximumTranscriptBytes) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " has an empty or overlong transcript");
  }

  const std::vector<std::string_view> raw_actions =
      split_preserving_empty(fields[3], '/');
  if (raw_actions.empty() || raw_actions.size() > kMaximumTurns) {
    throw std::invalid_argument("line " + std::to_string(line_number) +
                                " has an invalid number of turns");
  }

  GameRecord result{std::string(fields[0]), candidate_player, winner, {}, {}};
  result.actions.reserve(raw_actions.size());
  for (const std::string_view action : raw_actions) {
    if (action.empty() || action.size() > kMaximumActionBytes ||
        !std::all_of(action.begin(), action.end(),
                     [](char direction) {
                       return direction >= '0' && direction <= '7';
                     })) {
      throw std::invalid_argument("line " + std::to_string(line_number) +
                                  " contains an invalid complete-turn action");
    }
    result.actions.emplace_back(action);
  }
  return result;
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
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
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
  if (!saw_header) {
    throw std::invalid_argument("missing TSV input header");
  }
  if (records.empty()) {
    throw std::invalid_argument("input contains no games");
  }
  return records;
}

void validate_record(const GameRecord &record) {
  GameState state = make_initial_state(codingame_rules());
  for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
    if (is_terminal(state)) {
      throw std::invalid_argument("game " + record.game_id +
                                  " contains a turn after terminal state");
    }
    try {
      audit_reference::apply_encoded_turn(state, record.actions[turn]);
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
                                " winner does not match replayed state");
  }
}

void validate_records(const std::vector<GameRecord> &records) {
  for (const GameRecord &record : records) {
    validate_record(record);
  }
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
    if (left.a == right.a) {
      return left.b < right.b;
    }
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

std::uint32_t decision_time_limit(std::uint32_t first_time_ms,
                                  std::uint32_t later_time_ms,
                                  std::size_t own_decision_index) noexcept {
  return own_decision_index == 0U ? first_time_ms : later_time_ms;
}

ReplayCorrection inspect_replay_correction(const GameState &state,
                                           int player,
                                           std::string_view transcript) {
  ReplayCorrection correction;
  correction.lookup_found = audit_candidate::lookup_replay_correction(
      player, transcript, correction.lookup_action);
  if (!correction.lookup_found) {
    return correction;
  }

  GameState replay_state = state;
  correction.valid = audit_candidate::try_replay_correction(
      replay_state, player, transcript, correction.action);
  if (!correction.valid) {
    correction.action.clear();
  }
  return correction;
}

std::string reconstruct_deployed_action(
    std::string_view raw_bfm_action,
    const ReplayCorrection &replay_correction) {
  return replay_correction.valid ? replay_correction.action
                                 : std::string(raw_bfm_action);
}

template <typename EncodeDirection>
std::string encode_complete_action(const GameState &root,
                                   const std::vector<Move> &moves,
                                   EncodeDirection encode_direction) {
  if (moves.empty()) {
    throw std::logic_error("search returned an empty action");
  }
  GameState state = root;
  const Player mover = root.to_move;
  std::string encoded;
  encoded.reserve(moves.size());
  for (const Move move : moves) {
    if (is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("search action continues after turn end");
    }
    encoded.push_back(encode_direction(state.ball, move.to));
    state = apply_move(state, move);
  }
  if (!is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("search action omits a required rebound");
  }
  return encoded;
}

CandidateDecision choose_candidate(const GameState &state,
                                   const AuditConfig &audit_config,
                                   std::uint32_t time_limit_ms) {
  CandidateDecision decision;
  audit_candidate::SearchConfig config;
  config.max_work = audit_config.candidate_work;
  config.max_tree_nodes = audit_config.candidate_tree_nodes;
  config.max_actions = audit_config.candidate_max_actions;
  config.nonroot_actions = audit_config.candidate_nonroot_actions;
  config.max_partial_paths = audit_config.candidate_max_partial_paths;
  config.replay_value_blend_percent =
      audit_config.replay_value_blend_percent;
  config.teacher_residual_weight_percent =
      audit_config.teacher_residual_weight_percent;
  config.exploration_constant = audit_config.candidate_exploration;
  config.first_play_urgency = audit_config.candidate_fpu;
  config.final_visit_weight = audit_config.candidate_final_visit_weight;
  config.root_only = audit_config.candidate_root_only;

  const auto started = audit_candidate::SearchClock::now();
  if (time_limit_ms != 0) {
    config.absolute_deadline =
        started + std::chrono::milliseconds(time_limit_ms);
  }
  audit_candidate::FullTurnBfmSearch search(state, config);
  const std::vector<Move> moves = search.run();
  decision.elapsed_ms =
      std::chrono::duration<double, std::milli>(
          audit_candidate::SearchClock::now() - started)
          .count();
  decision.action = encode_complete_action(
      state, moves, audit_candidate::encode_direction);

  const audit_candidate::SearchStats &stats = search.stats();
  decision.work = stats.work;
  decision.tree_nodes = stats.tree_nodes;
  decision.expansions = stats.expansions;
  decision.child_evaluations = stats.child_evaluations;
  decision.generator_partial_paths = stats.generator_partial_paths;
  decision.completed_actions = stats.completed_actions;
  decision.generator_duplicates = stats.generator_duplicates;
  decision.tactical_actions = stats.tactical_actions;
  decision.generator_truncations = stats.generator_truncations;
  decision.root_score = stats.root_score;
  decision.budget_exhausted = stats.budget_exhausted;
  decision.deadline_reached = stats.deadline_reached;
  decision.node_cap_reached = stats.node_cap_reached;
  return decision;
}

ReferenceDecision choose_reference(const GameState &state,
                                   const AuditConfig &audit_config,
                                   std::uint32_t time_limit_ms) {
  ReferenceDecision decision;
  audit_reference::SearchConfig config;
  config.max_nodes = audit_config.reference_nodes;
  config.max_turn_depth = audit_config.reference_depth;
  config.replay_value_blend_percent = audit_config.replay_value_blend_percent;
  config.teacher_residual_weight_percent =
      audit_config.teacher_residual_weight_percent;

  const auto started = audit_reference::SearchClock::now();
  if (time_limit_ms != 0) {
    config.absolute_deadline =
        started + std::chrono::milliseconds(time_limit_ms);
  }
  audit_reference::CompleteTurnSearch search(state, config);
  const std::vector<Move> moves = search.run();
  decision.elapsed_ms =
      std::chrono::duration<double, std::milli>(
          audit_reference::SearchClock::now() - started)
          .count();
  decision.action = encode_complete_action(
      state, moves, audit_reference::encode_direction);

  const audit_reference::SearchStats &stats = search.stats();
  decision.nodes = stats.nodes;
  decision.completed_turn_depth = stats.completed_turn_depth;
  decision.attempted_turn_depth = stats.attempted_turn_depth;
  decision.root_score = stats.root_score;
  decision.budget_exhausted = stats.budget_exhausted;
  return decision;
}

std::string tactical_class_name(audit_candidate::TacticalClass tactical) {
  switch (tactical) {
    case audit_candidate::TacticalClass::ImmediateWin:
      return "immediate-win";
    case audit_candidate::TacticalClass::ForcedCutoff:
      return "forced-cutoff";
    case audit_candidate::TacticalClass::SafeHandoff:
      return "safe-handoff";
    case audit_candidate::TacticalClass::OpponentImmediateWin:
      return "opponent-immediate-win";
    case audit_candidate::TacticalClass::TerminalLoss:
      return "terminal-loss";
  }
  throw std::logic_error("unknown tactical class");
}

ps::detail::PositionKey action_boundary_key(
    const GameState &state, std::string_view action,
    const std::shared_ptr<const ps::detail::SearchTopology> &topology) {
  GameState endpoint = state;
  audit_candidate::apply_encoded_turn(endpoint, action);
  const ps::detail::SearchPosition compact(topology, endpoint);
  return audit_candidate::boundary_key(compact);
}

RetainedActionMatch find_retained_action(
    const audit_candidate::GenerationResult &generated,
    std::string_view encoded, ps::detail::PositionKey boundary) {
  RetainedActionMatch match;
  for (std::size_t index = 0; index < generated.actions.size(); ++index) {
    const auto &action = generated.actions[index];
    if (match.exact_ordinal < 0 && action.encoded == encoded) {
      match.exact_ordinal = static_cast<int>(index);
    }
    if (match.boundary_ordinal < 0 &&
        audit_candidate::boundary_key(action.result) == boundary) {
      match.boundary_ordinal = static_cast<int>(index);
      match.tactical_class = tactical_class_name(action.tactical);
    }
  }
  return match;
}

RootCoverage inspect_root_coverage(const GameState &state,
                                   const AuditConfig &config,
                                   std::string_view actual_action,
                                   std::string_view candidate_action,
                                   std::string_view reference_action) {
  auto topology = std::make_shared<ps::detail::SearchTopology>(state.config);
  const ps::detail::SearchPosition root(topology, state);
  audit_candidate::GeneratorConfig generator_config;
  generator_config.max_actions = config.candidate_max_actions;
  generator_config.max_partial_paths = config.candidate_max_partial_paths;
  generator_config.max_work = config.candidate_work;
  audit_candidate::FullTurnGenerator generator(topology, root,
                                                generator_config);
  const audit_candidate::GenerationResult generated = generator.run();
  RootCoverage coverage;
  coverage.actions = generated.actions.size();
  coverage.partial_paths = generated.stats.explored_partial_paths;
  coverage.completed_actions = generated.stats.completed_actions;
  coverage.duplicates = generated.stats.duplicates;
  coverage.truncations = generated.stats.truncations;
  coverage.exhaustive = generated.exhaustive;
  coverage.actual = find_retained_action(
      generated, actual_action,
      action_boundary_key(state, actual_action, topology));
  coverage.candidate = find_retained_action(
      generated, candidate_action,
      action_boundary_key(state, candidate_action, topology));
  coverage.reference = find_retained_action(
      generated, reference_action,
      action_boundary_key(state, reference_action, topology));
  return coverage;
}

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) {
    transcript.push_back('/');
  }
  transcript.append(action);
}

std::vector<DecisionAudit> audit_records(
    const std::vector<GameRecord> &records, const AuditConfig &config) {
  std::vector<DecisionAudit> result;
  for (const GameRecord &record : records) {
    GameState state = make_initial_state(codingame_rules());
    std::string prefix;
    std::size_t own_decision = 0;
    for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
      if (player_id(state.to_move) == record.candidate_player) {
        DecisionAudit audit;
        audit.game_id = record.game_id;
        audit.state_id = state_identity(state);
        audit.transcript_prefix = prefix;
        audit.turn_index = turn;
        audit.own_decision_index = own_decision++;
        audit.candidate_player = record.candidate_player;
        audit.winner = record.winner;
        audit.actual_action = record.actions[turn];
        audit.provenance = record.provenance;
        audit.candidate_time_limit_ms = decision_time_limit(
            config.candidate_first_time_ms, config.candidate_later_time_ms,
            audit.own_decision_index);
        audit.reference_time_limit_ms = decision_time_limit(
            config.reference_first_time_ms, config.reference_later_time_ms,
            audit.own_decision_index);
        audit.candidate =
            choose_candidate(state, config, audit.candidate_time_limit_ms);
        audit.reference =
            choose_reference(state, config, audit.reference_time_limit_ms);
        audit.replay_correction = inspect_replay_correction(
            state, record.candidate_player, prefix);
        audit.reconstructed_deployed_action = reconstruct_deployed_action(
            audit.candidate.action, audit.replay_correction);
        audit.root_coverage = inspect_root_coverage(
            state, config, audit.actual_action, audit.candidate.action,
            audit.reference.action);
        result.push_back(std::move(audit));
      }
      audit_reference::apply_encoded_turn(state, record.actions[turn]);
      append_turn(prefix, record.actions[turn]);
    }
  }
  return result;
}

std::string json_quote(std::string_view value) {
  std::string result;
  result.reserve(value.size() + 2U);
  result.push_back('"');
  constexpr char kHex[] = "0123456789abcdef";
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        result.append("\\\"");
        break;
      case '\\':
        result.append("\\\\");
        break;
      case '\b':
        result.append("\\b");
        break;
      case '\f':
        result.append("\\f");
        break;
      case '\n':
        result.append("\\n");
        break;
      case '\r':
        result.append("\\r");
        break;
      case '\t':
        result.append("\\t");
        break;
      default:
        if (character < 0x20U) {
          result.append("\\u00");
          result.push_back(kHex[character >> 4U]);
          result.push_back(kHex[character & 0x0fU]);
        } else {
          result.push_back(static_cast<char>(character));
        }
    }
  }
  result.push_back('"');
  return result;
}

std::string color_name(int candidate_player) {
  return candidate_player == 0 ? "player_one" : "player_two";
}

std::string result_name(const DecisionAudit &audit) {
  return audit.candidate_player == audit.winner ? "win" : "loss";
}

std::string provenance_json(const ProvenanceMetadata &provenance) {
  std::string result{"{"};
  for (std::size_t index = 0; index < provenance.size(); ++index) {
    if (index != 0U) {
      result.push_back(',');
    }
    result.append(json_quote(provenance[index].first));
    result.push_back(':');
    result.append(json_quote(provenance[index].second));
  }
  result.push_back('}');
  return result;
}

void write_json_line(std::ostream &output, const DecisionAudit &audit,
                     const AuditConfig &config) {
  output << '{'
         << "\"schema_version\":" << json_quote(kSchemaVersion) << ','
         << "\"input_provenance\":" << provenance_json(audit.provenance)
         << ','
         << "\"game_id\":" << json_quote(audit.game_id) << ','
         << "\"state_id\":" << json_quote(audit.state_id) << ','
         << "\"transcript_prefix\":"
         << json_quote(audit.transcript_prefix) << ','
         << "\"turn_index\":" << audit.turn_index << ','
         << "\"own_decision_index\":" << audit.own_decision_index << ','
         << "\"candidate_player\":" << audit.candidate_player << ','
         << "\"color\":" << json_quote(color_name(audit.candidate_player))
         << ',' << "\"winner\":" << audit.winner << ','
         << "\"result\":" << json_quote(result_name(audit)) << ','
         << "\"actual_action\":" << json_quote(audit.actual_action) << ','
         << "\"candidate_action\":" << json_quote(audit.candidate.action)
         << ',' << "\"reference_action\":"
         << json_quote(audit.reference.action) << ','
         << "\"replay_correction_lookup_found\":"
         << (audit.replay_correction.lookup_found ? "true" : "false") << ','
         << "\"replay_correction_lookup_action\":"
         << json_quote(audit.replay_correction.lookup_action) << ','
         << "\"replay_correction_valid\":"
         << (audit.replay_correction.valid ? "true" : "false") << ','
         << "\"replay_correction_action\":"
         << json_quote(audit.replay_correction.action) << ','
         << "\"reconstructed_deployed_action\":"
         << json_quote(audit.reconstructed_deployed_action) << ','
         << "\"candidate_matches_actual\":"
         << (audit.candidate.action == audit.actual_action ? "true" : "false")
         << ',' << "\"reference_matches_actual\":"
         << (audit.reference.action == audit.actual_action ? "true" : "false")
         << ',' << "\"candidate_matches_reference\":"
         << (audit.candidate.action == audit.reference.action ? "true"
                                                               : "false")
         << ',' << "\"reconstructed_deployed_matches_actual\":"
         << (audit.reconstructed_deployed_action == audit.actual_action
                 ? "true"
                 : "false")
         << ',' << "\"diagnostic_root_actions\":"
         << audit.root_coverage.actions
         << ',' << "\"diagnostic_root_exhaustive\":"
         << (audit.root_coverage.exhaustive ? "true" : "false")
         << ',' << "\"diagnostic_root_partial_paths\":"
         << audit.root_coverage.partial_paths
         << ',' << "\"diagnostic_root_completed_actions\":"
         << audit.root_coverage.completed_actions
         << ',' << "\"diagnostic_root_duplicates\":"
         << audit.root_coverage.duplicates
         << ',' << "\"diagnostic_root_truncations\":"
         << audit.root_coverage.truncations
         << ',' << "\"actual_action_retained_ordinal\":"
         << audit.root_coverage.actual.exact_ordinal
         << ',' << "\"actual_boundary_retained_ordinal\":"
         << audit.root_coverage.actual.boundary_ordinal
         << ',' << "\"actual_retained_tactical_class\":"
         << json_quote(audit.root_coverage.actual.tactical_class)
         << ',' << "\"candidate_action_retained_ordinal\":"
         << audit.root_coverage.candidate.exact_ordinal
         << ',' << "\"candidate_boundary_retained_ordinal\":"
         << audit.root_coverage.candidate.boundary_ordinal
         << ',' << "\"reference_action_retained_ordinal\":"
         << audit.root_coverage.reference.exact_ordinal
         << ',' << "\"reference_boundary_retained_ordinal\":"
         << audit.root_coverage.reference.boundary_ordinal
         << ',' << "\"reference_retained_tactical_class\":"
         << json_quote(audit.root_coverage.reference.tactical_class)
         << ',' << "\"candidate_work_limit\":" << config.candidate_work
         << ',' << "\"candidate_tree_node_limit\":"
         << config.candidate_tree_nodes
         << ',' << "\"candidate_max_actions\":"
         << config.candidate_max_actions
         << ',' << "\"candidate_nonroot_actions\":"
         << config.candidate_nonroot_actions
         << ',' << "\"candidate_max_partial_paths\":"
         << config.candidate_max_partial_paths
         << ',' << "\"candidate_exploration\":" << std::setprecision(9)
         << config.candidate_exploration
         << ',' << "\"candidate_fpu\":" << std::setprecision(9)
         << config.candidate_fpu
         << ',' << "\"candidate_final_visit_weight\":"
         << std::setprecision(9) << config.candidate_final_visit_weight
         << ',' << "\"candidate_root_only\":"
         << (config.candidate_root_only ? "true" : "false")
         << ',' << "\"candidate_first_time_limit_ms\":"
         << config.candidate_first_time_ms
         << ',' << "\"candidate_later_time_limit_ms\":"
         << config.candidate_later_time_ms
         << ',' << "\"candidate_time_limit_ms\":"
         << audit.candidate_time_limit_ms
         << ',' << "\"candidate_elapsed_ms\":"
         << std::setprecision(9) << audit.candidate.elapsed_ms << ','
         << "\"candidate_work\":" << audit.candidate.work << ','
         << "\"candidate_tree_nodes\":" << audit.candidate.tree_nodes << ','
         << "\"candidate_expansions\":" << audit.candidate.expansions << ','
         << "\"candidate_child_evaluations\":"
         << audit.candidate.child_evaluations << ','
         << "\"candidate_generator_partial_paths\":"
         << audit.candidate.generator_partial_paths << ','
         << "\"candidate_completed_actions\":"
         << audit.candidate.completed_actions << ','
         << "\"candidate_generator_duplicates\":"
         << audit.candidate.generator_duplicates << ','
         << "\"candidate_tactical_actions\":"
         << audit.candidate.tactical_actions << ','
         << "\"candidate_generator_truncations\":"
         << audit.candidate.generator_truncations << ','
         << "\"candidate_budget_exhausted\":"
         << (audit.candidate.budget_exhausted ? "true" : "false") << ','
         << "\"candidate_deadline_reached\":"
         << (audit.candidate.deadline_reached ? "true" : "false") << ','
         << "\"candidate_node_cap_reached\":"
         << (audit.candidate.node_cap_reached ? "true" : "false") << ','
         << "\"candidate_root_score\":" << audit.candidate.root_score << ','
         << "\"reference_nodes_limit\":" << config.reference_nodes << ','
         << "\"reference_depth_limit\":" << config.reference_depth << ','
         << "\"reference_first_time_limit_ms\":"
         << config.reference_first_time_ms << ','
         << "\"reference_later_time_limit_ms\":"
         << config.reference_later_time_ms << ','
         << "\"reference_time_limit_ms\":"
         << audit.reference_time_limit_ms << ','
         << "\"reference_elapsed_ms\":" << std::setprecision(9)
         << audit.reference.elapsed_ms << ',' << "\"reference_nodes\":"
         << audit.reference.nodes << ','
         << "\"reference_completed_turn_depth\":"
         << audit.reference.completed_turn_depth << ','
         << "\"reference_attempted_turn_depth\":"
         << audit.reference.attempted_turn_depth << ','
         << "\"reference_budget_exhausted\":"
         << (audit.reference.budget_exhausted ? "true" : "false") << ','
         << "\"reference_root_score\":" << audit.reference.root_score << ','
         << "\"value_blend_percent\":"
         << config.replay_value_blend_percent << ','
         << "\"teacher_residual_percent\":"
         << config.teacher_residual_weight_percent << ','
         << "\"codingame_clock_mode\":"
         << (config.codingame_clocks ? "true" : "false")
         << "}\n";
}

constexpr std::string_view kTsvHeader =
    "schema_version\tinput_provenance_json\tgame_id\tstate_id\t"
    "transcript_prefix\tturn_index\t"
    "own_decision_index\tcandidate_player\tcolor\twinner\tresult\t"
    "actual_action\tcandidate_action\treference_action\t"
    "replay_correction_lookup_found\treplay_correction_lookup_action\t"
    "replay_correction_valid\treplay_correction_action\t"
    "reconstructed_deployed_action\t"
    "candidate_matches_actual\treference_matches_actual\t"
    "candidate_matches_reference\treconstructed_deployed_matches_actual\t"
    "diagnostic_root_actions\tdiagnostic_root_exhaustive\t"
    "diagnostic_root_partial_paths\tdiagnostic_root_completed_actions\t"
    "diagnostic_root_duplicates\tdiagnostic_root_truncations\t"
    "actual_action_retained_ordinal\tactual_boundary_retained_ordinal\t"
    "actual_retained_tactical_class\t"
    "candidate_action_retained_ordinal\t"
    "candidate_boundary_retained_ordinal\t"
    "reference_action_retained_ordinal\t"
    "reference_boundary_retained_ordinal\t"
    "reference_retained_tactical_class\t"
    "candidate_work_limit\t"
    "candidate_tree_node_limit\tcandidate_max_actions\t"
    "candidate_nonroot_actions\t"
    "candidate_max_partial_paths\tcandidate_exploration\tcandidate_fpu\t"
    "candidate_final_visit_weight\tcandidate_root_only\t"
    "candidate_first_time_limit_ms\tcandidate_later_time_limit_ms\t"
    "candidate_time_limit_ms\tcandidate_elapsed_ms\tcandidate_work\t"
    "candidate_tree_nodes\tcandidate_expansions\t"
    "candidate_child_evaluations\tcandidate_generator_partial_paths\t"
    "candidate_completed_actions\tcandidate_generator_duplicates\t"
    "candidate_tactical_actions\tcandidate_generator_truncations\t"
    "candidate_budget_exhausted\tcandidate_deadline_reached\t"
    "candidate_node_cap_reached\tcandidate_root_score\t"
    "reference_nodes_limit\treference_depth_limit\t"
    "reference_first_time_limit_ms\treference_later_time_limit_ms\t"
    "reference_time_limit_ms\treference_elapsed_ms\t"
    "reference_nodes\treference_completed_turn_depth\t"
    "reference_attempted_turn_depth\treference_budget_exhausted\t"
    "reference_root_score\tvalue_blend_percent\tteacher_residual_percent\t"
    "codingame_clock_mode";

void write_tsv_line(std::ostream &output, const DecisionAudit &audit,
                    const AuditConfig &config) {
  output << kSchemaVersion << '\t' << provenance_json(audit.provenance) << '\t'
         << audit.game_id << '\t' << audit.state_id << '\t'
         << audit.transcript_prefix << '\t' << audit.turn_index << '\t'
         << audit.own_decision_index << '\t' << audit.candidate_player << '\t'
         << color_name(audit.candidate_player) << '\t' << audit.winner << '\t'
         << result_name(audit) << '\t' << audit.actual_action << '\t'
         << audit.candidate.action << '\t' << audit.reference.action << '\t'
         << audit.replay_correction.lookup_found << '\t'
         << audit.replay_correction.lookup_action << '\t'
         << audit.replay_correction.valid << '\t'
         << audit.replay_correction.action << '\t'
         << audit.reconstructed_deployed_action << '\t'
         << (audit.candidate.action == audit.actual_action) << '\t'
         << (audit.reference.action == audit.actual_action) << '\t'
         << (audit.candidate.action == audit.reference.action) << '\t'
         << (audit.reconstructed_deployed_action == audit.actual_action) << '\t'
         << audit.root_coverage.actions << '\t'
         << audit.root_coverage.exhaustive << '\t'
         << audit.root_coverage.partial_paths << '\t'
         << audit.root_coverage.completed_actions << '\t'
         << audit.root_coverage.duplicates << '\t'
         << audit.root_coverage.truncations << '\t'
         << audit.root_coverage.actual.exact_ordinal << '\t'
         << audit.root_coverage.actual.boundary_ordinal << '\t'
         << audit.root_coverage.actual.tactical_class << '\t'
         << audit.root_coverage.candidate.exact_ordinal << '\t'
         << audit.root_coverage.candidate.boundary_ordinal << '\t'
         << audit.root_coverage.reference.exact_ordinal << '\t'
         << audit.root_coverage.reference.boundary_ordinal << '\t'
         << audit.root_coverage.reference.tactical_class << '\t'
         << config.candidate_work << '\t' << config.candidate_tree_nodes << '\t'
         << config.candidate_max_actions << '\t'
         << config.candidate_nonroot_actions << '\t'
         << config.candidate_max_partial_paths << '\t' << std::setprecision(9)
         << config.candidate_exploration << '\t' << config.candidate_fpu << '\t'
         << config.candidate_final_visit_weight << '\t'
         << config.candidate_root_only << '\t'
         << config.candidate_first_time_ms << '\t'
         << config.candidate_later_time_ms << '\t'
         << audit.candidate_time_limit_ms << '\t'
         << std::setprecision(9) << audit.candidate.elapsed_ms << '\t'
         << audit.candidate.work << '\t' << audit.candidate.tree_nodes << '\t'
         << audit.candidate.expansions << '\t'
         << audit.candidate.child_evaluations << '\t'
         << audit.candidate.generator_partial_paths << '\t'
         << audit.candidate.completed_actions << '\t'
         << audit.candidate.generator_duplicates << '\t'
         << audit.candidate.tactical_actions << '\t'
         << audit.candidate.generator_truncations << '\t'
         << audit.candidate.budget_exhausted << '\t'
         << audit.candidate.deadline_reached << '\t'
         << audit.candidate.node_cap_reached << '\t'
         << audit.candidate.root_score << '\t' << config.reference_nodes << '\t'
         << config.reference_depth << '\t'
         << config.reference_first_time_ms << '\t'
         << config.reference_later_time_ms << '\t'
         << audit.reference_time_limit_ms << '\t'
         << std::setprecision(9)
         << audit.reference.elapsed_ms << '\t' << audit.reference.nodes << '\t'
         << audit.reference.completed_turn_depth << '\t'
         << audit.reference.attempted_turn_depth << '\t'
         << audit.reference.budget_exhausted << '\t'
         << audit.reference.root_score << '\t'
         << config.replay_value_blend_percent << '\t'
         << config.teacher_residual_weight_percent << '\t'
         << config.codingame_clocks << '\n';
}

void write_audits(std::ostream &output,
                  const std::vector<DecisionAudit> &audits,
                  const AuditConfig &config) {
  if (config.output_format == OutputFormat::Tsv) {
    output << kTsvHeader << '\n';
  }
  for (const DecisionAudit &audit : audits) {
    if (config.output_format == OutputFormat::JsonLines) {
      write_json_line(output, audit, config);
    } else {
      write_tsv_line(output, audit, config);
    }
  }
}

void validate_config(const AuditConfig &config) {
  if (config.candidate_work == 0 ||
      config.candidate_work > audit_candidate::kMaximumWork ||
      config.candidate_tree_nodes < 2 ||
      config.candidate_tree_nodes > audit_candidate::kMaximumTreeNodes ||
      config.candidate_max_actions == 0 ||
      config.candidate_max_actions > audit_candidate::kMaximumActions ||
      config.candidate_nonroot_actions > audit_candidate::kMaximumActions ||
      config.candidate_max_partial_paths == 0 ||
      config.candidate_max_partial_paths >
          audit_candidate::kMaximumPartialPaths ||
      !std::isfinite(config.candidate_exploration) ||
      config.candidate_exploration < 0.0 ||
      !std::isfinite(config.candidate_fpu) ||
      !std::isfinite(config.candidate_final_visit_weight) ||
      config.candidate_final_visit_weight < 0.0 ||
      config.reference_nodes == 0 ||
      config.reference_nodes > audit_reference::kMaximumNodes ||
      config.reference_depth == 0 ||
      config.reference_depth > audit_reference::kMaximumTurnDepth ||
      config.replay_value_blend_percent < 0 ||
      config.replay_value_blend_percent > 100 ||
      config.teacher_residual_weight_percent < 0 ||
      config.teacher_residual_weight_percent > 100) {
    throw std::invalid_argument("invalid replay decision audit configuration");
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
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (option == "--input") {
      config.input_path =
          std::string(require_value(index, argc, argv, option));
    } else if (option == "--format") {
      const std::string_view format = require_value(index, argc, argv, option);
      if (format == "jsonl") {
        config.output_format = OutputFormat::JsonLines;
      } else if (format == "tsv") {
        config.output_format = OutputFormat::Tsv;
      } else {
        throw std::invalid_argument("--format must be jsonl or tsv");
      }
    } else if (option == "--candidate-work") {
      config.candidate_work = parse_integer<std::uint64_t>(
          require_value(index, argc, argv, option), "candidate work");
    } else if (option == "--candidate-tree-nodes") {
      config.candidate_tree_nodes = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "candidate tree nodes");
    } else if (option == "--candidate-max-actions") {
      config.candidate_max_actions = parse_integer<std::size_t>(
          require_value(index, argc, argv, option), "candidate max actions");
    } else if (option == "--candidate-nonroot-actions") {
      config.candidate_nonroot_actions = parse_integer<std::size_t>(
          require_value(index, argc, argv, option),
          "candidate non-root actions");
    } else if (option == "--candidate-max-partial-paths") {
      config.candidate_max_partial_paths = parse_integer<std::uint64_t>(
          require_value(index, argc, argv, option),
          "candidate max partial paths");
    } else if (option == "--candidate-time-ms") {
      const std::uint32_t time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "candidate time");
      config.candidate_first_time_ms = time_ms;
      config.candidate_later_time_ms = time_ms;
    } else if (option == "--candidate-first-ms") {
      config.candidate_first_time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "candidate first time");
    } else if (option == "--candidate-later-ms") {
      config.candidate_later_time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "candidate later time");
    } else if (option == "--candidate-exploration") {
      config.candidate_exploration = parse_double(
          require_value(index, argc, argv, option), "candidate exploration");
    } else if (option == "--candidate-fpu") {
      config.candidate_fpu = parse_double(
          require_value(index, argc, argv, option), "candidate FPU");
    } else if (option == "--candidate-final-visit-weight") {
      config.candidate_final_visit_weight = parse_double(
          require_value(index, argc, argv, option),
          "candidate final visit weight");
    } else if (option == "--candidate-root-only") {
      const unsigned int root_only = parse_integer<unsigned int>(
          require_value(index, argc, argv, option), "candidate root only");
      if (root_only > 1U) {
        throw std::invalid_argument("candidate root only must be 0 or 1");
      }
      config.candidate_root_only = root_only != 0U;
    } else if (option == "--reference-nodes") {
      config.reference_nodes = parse_integer<std::uint64_t>(
          require_value(index, argc, argv, option), "reference nodes");
    } else if (option == "--reference-depth") {
      config.reference_depth = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "reference depth");
    } else if (option == "--reference-time-ms") {
      const std::uint32_t time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "reference time");
      config.reference_first_time_ms = time_ms;
      config.reference_later_time_ms = time_ms;
    } else if (option == "--reference-first-ms") {
      config.reference_first_time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "reference first time");
    } else if (option == "--reference-later-ms") {
      config.reference_later_time_ms = parse_integer<std::uint32_t>(
          require_value(index, argc, argv, option), "reference later time");
    } else if (option == "--codingame-clocks") {
      config.codingame_clocks = true;
    } else if (option == "--value-blend-percent") {
      config.replay_value_blend_percent = parse_integer<int>(
          require_value(index, argc, argv, option), "value blend percent");
    } else if (option == "--teacher-residual-percent") {
      config.teacher_residual_weight_percent = parse_integer<int>(
          require_value(index, argc, argv, option),
          "teacher residual percent");
    } else if (option == "--help") {
      throw std::invalid_argument("help");
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (config.codingame_clocks) {
    config.candidate_work = audit_candidate::kMaximumWork;
    config.reference_nodes = audit_reference::kMaximumNodes;
    config.candidate_first_time_ms = audit_candidate::kFirstSearchTimeMs;
    config.candidate_later_time_ms = audit_candidate::kLaterSearchTimeMs;
    config.reference_first_time_ms = audit_reference::kFirstSearchTimeMs;
    config.reference_later_time_ms = audit_reference::kLaterSearchTimeMs;
  }
  validate_config(config);
  return config;
}

void write_usage(std::ostream &output, std::string_view executable) {
  output
      << "usage: " << executable << " [options] < games.tsv\n"
      << "input header: " << kInputHeader << "\n"
      << "options:\n"
      << "  --input PATH\n"
      << "  --format jsonl|tsv\n"
      << "  --candidate-work N\n"
      << "  --candidate-tree-nodes N\n"
      << "  --candidate-max-actions N\n"
      << "  --candidate-nonroot-actions N (0 reuses root cap)\n"
      << "  --candidate-max-partial-paths N\n"
      << "  --candidate-time-ms N       set both decision clocks (0 disables)\n"
      << "  --candidate-first-ms N\n"
      << "  --candidate-later-ms N\n"
      << "  --candidate-exploration X\n"
      << "  --candidate-fpu X\n"
      << "  --candidate-final-visit-weight X\n"
      << "  --candidate-root-only 0|1\n"
      << "  --reference-nodes N\n"
      << "  --reference-depth N\n"
      << "  --reference-time-ms N       set both decision clocks (0 disables)\n"
      << "  --reference-first-ms N\n"
      << "  --reference-later-ms N\n"
      << "  --codingame-clocks          800/165 ms and production 3m caps\n"
      << "  --value-blend-percent N\n"
      << "  --teacher-residual-percent N\n";
}

int run(int argc, char **argv, std::istream &standard_input,
        std::ostream &output, std::ostream &error) {
  try {
    const AuditConfig config = parse_arguments(argc, argv);
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
    const std::vector<DecisionAudit> audits = audit_records(records, config);
    write_audits(output, audits, config);
    return 0;
  } catch (const std::invalid_argument &exception) {
    if (std::string_view(exception.what()) == "help") {
      write_usage(output, argc > 0 ? argv[0] : "replay_decision_auditor");
      return 0;
    }
    error << "error: " << exception.what() << '\n';
    return 2;
  } catch (const std::exception &exception) {
    error << "error: " << exception.what() << '\n';
    return 1;
  }
}

}  // namespace papersoccer::fullturn_replay_audit

#ifndef PAPER_SOCCER_REPLAY_DECISION_AUDITOR_NO_MAIN
int main(int argc, char **argv) {
  return papersoccer::fullturn_replay_audit::run(
      argc, argv, std::cin, std::cout, std::cerr);
}
#endif
