#define PAPER_SOCCER_REPLAY_DECISION_AUDITOR_NO_MAIN
#include "replay_decision_auditor.cpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace audit = papersoccer::fullturn_replay_audit;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

template <typename Function>
void require_invalid_argument(Function &&function, std::string_view message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

std::vector<audit::GameRecord> read(std::string_view contents) {
  std::istringstream input{std::string(contents)};
  return audit::read_records(input);
}

audit::AuditConfig fast_config() {
  audit::AuditConfig config;
  config.candidate_work = 256;
  config.candidate_tree_nodes = 256;
  config.candidate_max_actions = 16;
  config.candidate_max_partial_paths = 128;
  config.reference_nodes = 128;
  config.reference_depth = 4;
  return config;
}

audit::AuditConfig parse_options(std::vector<std::string> options) {
  options.insert(options.begin(), "replay_decision_auditor");
  std::vector<char *> arguments;
  arguments.reserve(options.size());
  for (std::string &option : options) {
    arguments.push_back(option.data());
  }
  return audit::parse_arguments(static_cast<int>(arguments.size()),
                                arguments.data());
}

class JsonValidator {
 public:
  explicit JsonValidator(std::string_view input) : input_(input) {}

  bool valid_document() {
    skip_space();
    if (!parse_value()) {
      return false;
    }
    skip_space();
    return offset_ == input_.size();
  }

 private:
  void skip_space() {
    while (offset_ < input_.size() &&
           (input_[offset_] == ' ' || input_[offset_] == '\n' ||
            input_[offset_] == '\r' || input_[offset_] == '\t')) {
      ++offset_;
    }
  }

  bool consume(char expected) {
    skip_space();
    if (offset_ >= input_.size() || input_[offset_] != expected) {
      return false;
    }
    ++offset_;
    return true;
  }

  bool parse_string(std::string *decoded = nullptr) {
    skip_space();
    if (offset_ >= input_.size() || input_[offset_++] != '"') {
      return false;
    }
    while (offset_ < input_.size()) {
      const unsigned char character =
          static_cast<unsigned char>(input_[offset_++]);
      if (character == '"') {
        return true;
      }
      if (character < 0x20U) {
        return false;
      }
      if (character != '\\') {
        if (decoded != nullptr) {
          decoded->push_back(static_cast<char>(character));
        }
        continue;
      }
      if (offset_ >= input_.size()) {
        return false;
      }
      const char escape = input_[offset_++];
      if (std::string_view{"\"\\/bfnrt"}.find(escape) !=
          std::string_view::npos) {
        if (decoded != nullptr) {
          decoded->push_back(escape);
        }
        continue;
      }
      if (escape != 'u' || offset_ + 4U > input_.size()) {
        return false;
      }
      for (int digit = 0; digit < 4; ++digit) {
        const char hexadecimal = input_[offset_++];
        if (!((hexadecimal >= '0' && hexadecimal <= '9') ||
              (hexadecimal >= 'a' && hexadecimal <= 'f') ||
              (hexadecimal >= 'A' && hexadecimal <= 'F'))) {
          return false;
        }
      }
      if (decoded != nullptr) {
        decoded->push_back('?');
      }
    }
    return false;
  }

  bool parse_number() {
    skip_space();
    const std::size_t begin = offset_;
    if (offset_ < input_.size() && input_[offset_] == '-') {
      ++offset_;
    }
    if (offset_ >= input_.size()) {
      return false;
    }
    if (input_[offset_] == '0') {
      ++offset_;
    } else if (input_[offset_] >= '1' && input_[offset_] <= '9') {
      while (offset_ < input_.size() && input_[offset_] >= '0' &&
             input_[offset_] <= '9') {
        ++offset_;
      }
    } else {
      return false;
    }
    if (offset_ < input_.size() && input_[offset_] == '.') {
      ++offset_;
      const std::size_t fraction = offset_;
      while (offset_ < input_.size() && input_[offset_] >= '0' &&
             input_[offset_] <= '9') {
        ++offset_;
      }
      if (offset_ == fraction) {
        return false;
      }
    }
    if (offset_ < input_.size() &&
        (input_[offset_] == 'e' || input_[offset_] == 'E')) {
      ++offset_;
      if (offset_ < input_.size() &&
          (input_[offset_] == '+' || input_[offset_] == '-')) {
        ++offset_;
      }
      const std::size_t exponent = offset_;
      while (offset_ < input_.size() && input_[offset_] >= '0' &&
             input_[offset_] <= '9') {
        ++offset_;
      }
      if (offset_ == exponent) {
        return false;
      }
    }
    return offset_ > begin;
  }

  bool parse_object() {
    if (!consume('{')) {
      return false;
    }
    skip_space();
    if (offset_ < input_.size() && input_[offset_] == '}') {
      ++offset_;
      return true;
    }
    std::unordered_set<std::string> keys;
    while (true) {
      std::string key;
      if (!parse_string(&key) || !keys.insert(key).second || !consume(':') ||
          !parse_value()) {
        return false;
      }
      skip_space();
      if (offset_ < input_.size() && input_[offset_] == '}') {
        ++offset_;
        return true;
      }
      if (!consume(',')) {
        return false;
      }
    }
  }

  bool parse_value() {
    skip_space();
    if (offset_ >= input_.size()) {
      return false;
    }
    if (input_[offset_] == '{') {
      return parse_object();
    }
    if (input_[offset_] == '"') {
      return parse_string();
    }
    for (const std::string_view literal : {"true", "false", "null"}) {
      if (input_.substr(offset_).starts_with(literal)) {
        offset_ += literal.size();
        return true;
      }
    }
    return parse_number();
  }

  std::string_view input_;
  std::size_t offset_{};
};

std::vector<std::string_view> nonempty_lines(const std::string &text) {
  std::vector<std::string_view> lines;
  std::string_view remaining(text);
  while (!remaining.empty()) {
    const std::size_t newline = remaining.find('\n');
    const std::string_view line = remaining.substr(0, newline);
    if (!line.empty()) {
      lines.push_back(line);
    }
    if (newline == std::string_view::npos) {
      break;
    }
    remaining.remove_prefix(newline + 1U);
  }
  return lines;
}

void strict_input_parser_accepts_only_the_documented_shape() {
  const std::vector<audit::GameRecord> records = read(
      "# synthetic complete-turn fixture\n"
      "# agent_id=6606663\n"
      "# source_binding_status=asserted-not-api-verified\n"
      "game_id\tcandidate_player\twinner\tturns\n"
      "own-goal-1\t0\t0\t0/0/0/0/0/0\n");
  require(records.size() == 1, "Expected one parsed game.");
  require(records[0].game_id == "own-goal-1" &&
              records[0].candidate_player == 0 && records[0].winner == 0 &&
              records[0].actions.size() == 6 &&
              records[0].provenance.size() == 2 &&
              records[0].provenance[0].first == "agent_id" &&
              records[0].provenance[0].second == "6606663",
          "Parsed fields do not match the fixture.");

  require_invalid_argument(
      [] {
        (void)read("game_id\tcandidate_player\twinner\tturns\n"
                   "bad id\t0\t0\t0/0/0/0/0/0\n");
      },
      "Unsafe game IDs must be rejected.");
  require_invalid_argument(
      [] {
        (void)read("game_id\tcandidate_player\twinner\tturns\n"
                   "bad-direction\t0\t0\t0/0/x\n");
      },
      "Non-direction characters must be rejected.");
  require_invalid_argument(
      [] {
        (void)read("game_id\tcandidate_player\twinner\tturns\n"
                   "extra-field\t0\t0\t0/0\textra\n");
      },
      "Unexpected TSV fields must be rejected.");
  require_invalid_argument(
      [] {
        (void)read("# source_sha256=first\n"
                   "# source_sha256=second\n"
                   "game_id\tcandidate_player\twinner\tturns\n"
                   "duplicate-metadata\t0\t0\t0/0/0/0/0/0\n");
      },
      "Duplicate provenance keys must be rejected.");
}

void exact_codingame_rules_validate_terminal_transcripts() {
  const papersoccer::RulesConfig rules = audit::codingame_rules();
  require(rules.width == 8 && rules.height == 10 &&
              rules.goal_rule == papersoccer::GoalRule::OwnGoalsAllowed &&
              rules.blocked_rule == papersoccer::BlockedRule::MoverLoses,
          "The auditor must use CodinGame's exact rules.");

  const std::vector<audit::GameRecord> valid = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "own-goal-1\t0\t0\t0/0/0/0/0/0\n");
  audit::validate_records(valid);

  const std::vector<audit::GameRecord> nonterminal = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "nonterminal\t0\t0\t0\n");
  require_invalid_argument(
      [&] { audit::validate_records(nonterminal); },
      "A nonterminal transcript must be rejected.");

  const std::vector<audit::GameRecord> overlong = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "overlong\t0\t0\t0/0/0/0/0/00\n");
  require_invalid_argument(
      [&] { audit::validate_records(overlong); },
      "An action continuing after a goal must be rejected.");

  const std::vector<audit::GameRecord> wrong_winner = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "wrong-winner\t0\t1\t0/0/0/0/0/0\n");
  require_invalid_argument(
      [&] { audit::validate_records(wrong_winner); },
      "Declared winner must match exact replay state.");
}

void clock_modes_select_limits_by_own_decision_index() {
  require(audit::decision_time_limit(800, 165, 0) == 800 &&
              audit::decision_time_limit(800, 165, 1) == 165 &&
              audit::decision_time_limit(800, 165, 99) == 165,
          "Only the first own decision may use the first-turn clock.");

  const audit::AuditConfig explicit_clocks = parse_options(
      {"--candidate-first-ms", "701", "--candidate-later-ms", "151",
       "--reference-first-ms", "702", "--reference-later-ms", "152"});
  require(explicit_clocks.candidate_first_time_ms == 701 &&
              explicit_clocks.candidate_later_time_ms == 151 &&
              explicit_clocks.reference_first_time_ms == 702 &&
              explicit_clocks.reference_later_time_ms == 152 &&
              !explicit_clocks.codingame_clocks,
          "Explicit first/later clock flags were not retained.");

  const audit::AuditConfig codingame = parse_options(
      {"--candidate-work", "1", "--reference-nodes", "1",
       "--candidate-first-ms", "1", "--codingame-clocks"});
  require(codingame.codingame_clocks &&
              codingame.candidate_first_time_ms == 800 &&
              codingame.candidate_later_time_ms == 165 &&
              codingame.reference_first_time_ms == 800 &&
              codingame.reference_later_time_ms == 165 &&
              codingame.candidate_work == audit_candidate::kMaximumWork &&
              codingame.reference_nodes == audit_reference::kMaximumNodes &&
              codingame.candidate_work == 3'000'000 &&
              codingame.reference_nodes == 3'000'000,
          "CodinGame convenience mode must restore 800/165 and 3m caps.");

  std::vector<audit::GameRecord> records = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "clock-selection\t0\t0\t0/0/0/0/0/0\n");
  audit::AuditConfig config = fast_config();
  config.candidate_first_time_ms = 1'000;
  config.candidate_later_time_ms = 900;
  config.reference_first_time_ms = 800;
  config.reference_later_time_ms = 700;
  const std::vector<audit::DecisionAudit> decisions =
      audit::audit_records(records, config);
  require(decisions.size() == 3 &&
              decisions[0].candidate_time_limit_ms == 1'000 &&
              decisions[1].candidate_time_limit_ms == 900 &&
              decisions[2].candidate_time_limit_ms == 900 &&
              decisions[0].reference_time_limit_ms == 800 &&
              decisions[1].reference_time_limit_ms == 700,
          "Audit rows must report the clock actually used for that decision.");
}

void replay_book_path_reconstructs_the_deployed_action() {
  const audit_candidate::replay_book::Replay &replay =
      audit_candidate::replay_book::kReplays.front();
  const std::vector<std::string_view> actions =
      audit::split_preserving_empty(replay.transcript, '/');
  require(replay.first_turn < actions.size(),
          "Replay fixture needs an action at its correction turn.");

  papersoccer::GameState state =
      papersoccer::make_initial_state(audit::codingame_rules());
  std::string prefix;
  for (std::size_t turn = 0; turn < replay.first_turn; ++turn) {
    audit_reference::apply_encoded_turn(state, actions[turn]);
    audit::append_turn(prefix, actions[turn]);
  }
  require(audit::player_id(state.to_move) == replay.player_id,
          "Replay fixture correction must belong to the recorded player.");

  const audit::ReplayCorrection correction = audit::inspect_replay_correction(
      state, replay.player_id, prefix);
  require(correction.lookup_found && correction.valid &&
              correction.lookup_action == actions[replay.first_turn] &&
              correction.action == actions[replay.first_turn],
          "A legal replay-book correction must be exposed exactly.");
  require(audit::reconstruct_deployed_action("7", correction) ==
              actions[replay.first_turn],
          "A valid replay correction must override the raw BFM action.");

  const audit::ReplayCorrection none = audit::inspect_replay_correction(
      papersoccer::make_initial_state(audit::codingame_rules()), 0,
      "unmatched-prefix");
  require(!none.lookup_found && !none.valid &&
              audit::reconstruct_deployed_action("3", none) == "3",
          "Without a correction, the deployed reconstruction must use BFM.");
}

void initial_evaluation_reproduces_root_only_and_terminal_proofs() {
  std::vector<audit::GameRecord> records = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "initial-eval\t1\t0\t0/0/0/0/0/0\n");
  audit::validate_records(records);
  audit::AuditConfig config = fast_config();
  config.candidate_root_only = true;
  const std::vector<audit::DecisionAudit> decisions =
      audit::audit_records(records, config);
  require(decisions.size() == 3,
          "Player Two must have three audited decisions in the fixture.");
  for (const audit::DecisionAudit &decision : decisions) {
    const audit::InitialEvaluationDiagnostics &initial =
        decision.root_coverage.initial_evaluation;
    require(initial.best_retained_ordinal >= 0 &&
                initial.best_retained_ordinal <
                    static_cast<int>(decision.root_coverage.actions) &&
                !initial.best_action.empty(),
            "Every generated root needs a deterministic initial best action.");
    require(decision.candidate.action == initial.best_action &&
                decision.candidate.root_score == initial.best_score &&
                initial.candidate.score.has_value() &&
                initial.candidate.rank == 1 &&
                initial.candidate_bfm_change_assessable &&
                !initial.candidate_bfm_changed_from_initial_best,
            "Root-only search must reproduce the diagnostic initial ordering.");
  }

  const audit::DecisionAudit &terminal_decision = decisions.back();
  require(terminal_decision.actual_action == "0" &&
              terminal_decision.root_coverage.actual.tactical_class ==
                  "terminal-loss" &&
              terminal_decision.root_coverage.initial_evaluation.actual.score ==
                  audit_candidate::kMateScore - 1,
          "A retained terminal action must use the search's depth-one proof score.");

  papersoccer::GameState residual_state =
      papersoccer::make_initial_state(audit::codingame_rules());
  const std::vector<std::string_view> replay_actions =
      audit::split_preserving_empty(
          audit_candidate::replay_book::kReplays.front().transcript, '/');
  require(replay_actions.size() > 8,
          "Replay fixture must contain a residual-enabled prefix.");
  for (std::size_t turn = 0; turn < 8; ++turn) {
    audit_reference::apply_encoded_turn(residual_state, replay_actions[turn]);
  }
  require(!papersoccer::is_terminal(residual_state) &&
              residual_state.used_segments.size() >=
                  static_cast<std::size_t>(
                      audit_candidate::kTeacherResidualMinimumUsedEdges),
          "Replay prefix must enable the root-scoped teacher residual.");
  const audit::CandidateDecision residual_root_only =
      audit::choose_candidate(residual_state, config, 0);
  const audit::RootCoverage residual_coverage = audit::inspect_root_coverage(
      residual_state, config, residual_root_only.action,
      residual_root_only.action, residual_root_only.action);
  require(residual_coverage.initial_evaluation.best_action ==
                  residual_root_only.action &&
              residual_coverage.initial_evaluation.best_score ==
                  residual_root_only.root_score,
          "Initial evaluation must reproduce a residual-enabled root-only score.");
}

void fixed_work_audit_is_reproducible_and_complete() {
  const std::vector<audit::GameRecord> records = read(
      "# agent_id=6606663\n"
      "# source_binding_status=asserted-not-api-verified\n"
      "game_id\tcandidate_player\twinner\tturns\n"
      "own-goal-1\t0\t0\t0/0/0/0/0/0\n");
  audit::validate_records(records);
  const audit::AuditConfig config = fast_config();
  const std::vector<audit::DecisionAudit> first =
      audit::audit_records(records, config);
  const std::vector<audit::DecisionAudit> second =
      audit::audit_records(records, config);

  require(first.size() == 3 && second.size() == first.size(),
          "Player One must have three audited decisions in the fixture.");
  for (std::size_t index = 0; index < first.size(); ++index) {
    const audit::DecisionAudit &left = first[index];
    const audit::DecisionAudit &right = second[index];
    require(left.turn_index == index * 2U &&
                left.own_decision_index == index && left.actual_action == "0",
            "Decision indices or actual action are incorrect.");
    require(left.state_id.starts_with("fnv1a64:") &&
                left.state_id.size() == 24,
            "State identity must be a stable fixed-width digest.");
    require(left.candidate.action == right.candidate.action &&
                left.candidate.work == right.candidate.work &&
                left.candidate.tree_nodes == right.candidate.tree_nodes &&
                left.candidate.expansions == right.candidate.expansions &&
                left.candidate.child_evaluations ==
                    right.candidate.child_evaluations &&
                left.candidate.generator_partial_paths ==
                    right.candidate.generator_partial_paths &&
                left.candidate.completed_actions ==
                    right.candidate.completed_actions &&
                left.candidate.generator_duplicates ==
                    right.candidate.generator_duplicates &&
                left.candidate.tactical_actions ==
                    right.candidate.tactical_actions &&
                left.candidate.generator_truncations ==
                    right.candidate.generator_truncations &&
                left.candidate.deadline_reached ==
                    right.candidate.deadline_reached &&
                left.candidate.node_cap_reached ==
                    right.candidate.node_cap_reached &&
                left.candidate.root_score == right.candidate.root_score,
            "Fixed-work candidate decisions and counters must reproduce.");
    require(left.reference.action == right.reference.action &&
                left.reference.nodes == right.reference.nodes &&
                left.reference.root_score == right.reference.root_score,
            "Fixed-work reference decisions must reproduce.");
    require(!left.candidate.deadline_reached &&
                left.candidate_time_limit_ms == 0 &&
                left.reference_time_limit_ms == 0,
            "A zero time limit must leave the deadline disabled.");
    require(left.provenance == records[0].provenance &&
                left.reconstructed_deployed_action == left.candidate.action,
            "Provenance and the no-replay deployed path must be retained.");
    require(left.root_coverage.actions == right.root_coverage.actions &&
                left.root_coverage.partial_paths ==
                    right.root_coverage.partial_paths &&
                left.root_coverage.actual.exact_ordinal >= 0 &&
                left.root_coverage.actual.boundary_ordinal >= 0 &&
                left.root_coverage.candidate.boundary_ordinal >= 0,
            "Deterministic root coverage must locate actual and BFM actions.");
    const audit::InitialEvaluationDiagnostics &left_initial =
        left.root_coverage.initial_evaluation;
    const audit::InitialEvaluationDiagnostics &right_initial =
        right.root_coverage.initial_evaluation;
    require(left_initial.best_action == right_initial.best_action &&
                left_initial.best_score == right_initial.best_score &&
                left_initial.best_retained_ordinal ==
                    right_initial.best_retained_ordinal &&
                left_initial.actual.score == right_initial.actual.score &&
                left_initial.actual.rank == right_initial.actual.rank &&
                left_initial.candidate.score ==
                    right_initial.candidate.score &&
                left_initial.candidate.rank ==
                    right_initial.candidate.rank &&
                left_initial.reference.score ==
                    right_initial.reference.score &&
                left_initial.reference.rank ==
                    right_initial.reference.rank,
            "Initial action scores and ranks must reproduce exactly.");
    require(left_initial.actual.score.has_value() &&
                left_initial.actual.rank >= 1 &&
                left_initial.candidate.score.has_value() &&
                left_initial.candidate.rank >= 1 &&
                left_initial.candidate_bfm_change_assessable &&
                left_initial.candidate_bfm_changed_from_initial_best ==
                    (left.root_coverage.candidate.boundary_ordinal !=
                     left_initial.best_retained_ordinal),
            "Retained matches must expose consistent scores, ranks, and BFM change state.");
  }
  require(first[0].state_id != first[1].state_id &&
              first[1].state_id != first[2].state_id,
          "Distinct replay states need distinct identities.");

  std::ostringstream json;
  audit::write_audits(json, first, config);
  const std::string json_text = json.str();
  for (const std::string_view required : {
           "\"schema_version\":\"fullturn-decision-audit-v3\"",
           "\"input_provenance\":{\"agent_id\":\"6606663\"",
           "\"transcript_prefix\"", "\"actual_action\"",
           "\"candidate_action\"", "\"reference_action\"",
           "\"replay_correction_lookup_found\"",
           "\"replay_correction_valid\"",
           "\"reconstructed_deployed_action\"",
           "\"reconstructed_deployed_matches_actual\"",
           "\"candidate_work_limit\"", "\"candidate_tree_node_limit\"",
           "\"candidate_max_actions\"", "\"candidate_exploration\"",
           "\"candidate_fpu\"", "\"candidate_first_time_limit_ms\"",
           "\"candidate_final_visit_weight\"",
           "\"diagnostic_root_actions\"",
           "\"actual_action_retained_ordinal\"",
           "\"reference_boundary_retained_ordinal\"",
           "\"initial_eval_best_action\"",
           "\"initial_eval_best_score\"",
           "\"initial_eval_best_retained_ordinal\"",
           "\"actual_initial_eval_score\"",
           "\"actual_initial_eval_rank\"",
           "\"candidate_initial_eval_score\"",
           "\"candidate_initial_eval_rank\"",
           "\"reference_initial_eval_score\"",
           "\"reference_initial_eval_rank\"",
           "\"candidate_bfm_change_assessable\"",
           "\"candidate_bfm_changed_from_initial_best\"",
           "\"candidate_later_time_limit_ms\"",
           "\"candidate_time_limit_ms\"", "\"candidate_work\"",
           "\"candidate_tree_nodes\"",
           "\"candidate_expansions\"",
           "\"candidate_child_evaluations\"",
           "\"candidate_generator_partial_paths\"",
           "\"candidate_completed_actions\"",
           "\"candidate_generator_duplicates\"",
           "\"candidate_tactical_actions\"",
           "\"candidate_generator_truncations\"",
           "\"candidate_deadline_reached\"",
           "\"candidate_node_cap_reached\"",
           "\"candidate_root_score\"", "\"reference_first_time_limit_ms\"",
           "\"reference_later_time_limit_ms\"",
           "\"reference_time_limit_ms\"", "\"result\"", "\"color\""}) {
    require(json_text.find(required) != std::string::npos,
            "JSON output is missing a required decision field.");
  }
  const std::vector<std::string_view> json_lines = nonempty_lines(json_text);
  require(json_lines.size() == first.size(),
          "JSONL output must have one row per decision.");
  for (const std::string_view line : json_lines) {
    require(JsonValidator(line).valid_document(),
            "Every JSONL row must be valid JSON with unique object keys.");
    require(line.find("asserted-not-api-verified") != std::string_view::npos,
            "Every JSONL row must preserve collector provenance.");
  }

  audit::AuditConfig tsv_config = config;
  tsv_config.output_format = audit::OutputFormat::Tsv;
  std::ostringstream tsv;
  audit::write_audits(tsv, first, tsv_config);
  const std::string tsv_text = tsv.str();
  require(tsv_text.starts_with(audit::kTsvHeader) &&
              static_cast<std::size_t>(
                  std::count(tsv_text.begin(), tsv_text.end(), '\n')) ==
                  first.size() + 1U,
          "TSV output must have one header and one row per decision.");
  const std::vector<std::string_view> tsv_lines = nonempty_lines(tsv_text);
  for (std::size_t index = 1; index < tsv_lines.size(); ++index) {
    const std::string_view line = tsv_lines[index];
    require(std::count(line.begin(), line.end(), '\t') ==
                std::count(audit::kTsvHeader.begin(),
                           audit::kTsvHeader.end(), '\t'),
            "Every TSV row must align exactly with the schema header.");
    require(line.find("{\"agent_id\":\"6606663\"") !=
                std::string_view::npos &&
                line.find("asserted-not-api-verified") !=
                    std::string_view::npos,
            "Every TSV row must preserve collector provenance.");
  }
}

void command_validates_all_games_before_emitting_rows() {
  std::istringstream input(
      "game_id\tcandidate_player\twinner\tturns\n"
      "valid\t0\t0\t0/0/0/0/0/0\n"
      "invalid\t0\t0\t0\n");
  std::ostringstream output;
  std::ostringstream error;
  char executable[] = "replay_decision_auditor";
  char *arguments[] = {executable};
  const int status = audit::run(1, arguments, input, output, error);
  require(status == 2 && output.str().empty() &&
              error.str().find("not terminal") != std::string::npos,
          "Invalid batches must fail before any partial structured output.");
}

}  // namespace

int main() {
  const std::vector<std::pair<std::string_view, void (*)()>> tests{
      {"strict_input_parser_accepts_only_the_documented_shape",
       strict_input_parser_accepts_only_the_documented_shape},
      {"exact_codingame_rules_validate_terminal_transcripts",
       exact_codingame_rules_validate_terminal_transcripts},
      {"clock_modes_select_limits_by_own_decision_index",
       clock_modes_select_limits_by_own_decision_index},
      {"replay_book_path_reconstructs_the_deployed_action",
       replay_book_path_reconstructs_the_deployed_action},
      {"initial_evaluation_reproduces_root_only_and_terminal_proofs",
       initial_evaluation_reproduces_root_only_and_terminal_proofs},
      {"fixed_work_audit_is_reproducible_and_complete",
       fixed_work_audit_is_reproducible_and_complete},
      {"command_validates_all_games_before_emitting_rows",
       command_validates_all_games_before_emitting_rows},
  };

  int failures = 0;
  for (const auto &[name, test] : tests) {
    try {
      test();
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
  }
  return failures == 0 ? 0 : 1;
}
