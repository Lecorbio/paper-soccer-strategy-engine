#define PAPER_SOCCER_JACEK_NATIVE_REPLAY_AUDITOR_NO_MAIN
#include "replay_decision_auditor.cpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace audit = papersoccer::jacek_native_replay_audit;

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
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

std::vector<audit::GameRecord> read(std::string_view content) {
  std::istringstream input{std::string(content)};
  return audit::read_records(input);
}

audit::AuditConfig fast_config() {
  audit::AuditConfig config;
  config.fixed_work = 64;
  config.max_actions = 16;
  config.max_partial_paths = 64;
  return config;
}

audit::AuditConfig parse_options(std::vector<std::string> options) {
  options.insert(options.begin(), "replay_decision_auditor");
  std::vector<char *> arguments;
  arguments.reserve(options.size());
  for (std::string &option : options) arguments.push_back(option.data());
  return audit::parse_arguments(static_cast<int>(arguments.size()),
                                arguments.data());
}

std::size_t field_count(std::string_view line) {
  return 1U + static_cast<std::size_t>(
                  std::count(line.begin(), line.end(), '\t'));
}

void strict_collector_contract_and_provenance() {
  const std::vector<audit::GameRecord> records = read(
      "# agent_id=6600001\n"
      "# asserted_submission_id=41100001\n"
      "# source_binding_status=asserted-not-api-verified\n"
      "game_id\tcandidate_player\twinner\tturns\n"
      "synthetic\t0\t0\t0/0/0/0/0/0\n");
  require(records.size() == 1 && records[0].actions.size() == 6,
          "collector-compatible TSV must parse");
  require(records[0].provenance.size() == 3 &&
              records[0].provenance[0].first == "agent_id" &&
              records[0].provenance[0].second == "6600001",
          "collector provenance must remain attached to each game");

  require_invalid_argument(
      [] {
        (void)read("game_id\tcandidate_player\twinner\tturns\n"
                   "unsafe id\t0\t0\t0/0/0/0/0/0\n");
      },
      "unsafe game IDs must fail closed");
  require_invalid_argument(
      [] {
        (void)read("# source_sha256=one\n# source_sha256=two\n"
                   "game_id\tcandidate_player\twinner\tturns\n"
                   "duplicate\t0\t0\t0/0/0/0/0/0\n");
      },
      "duplicate provenance keys must fail closed");
  require_invalid_argument(
      [] {
        (void)read("game_id\tcandidate_player\twinner\tturns\n"
                   "direction\t0\t0\t0/0/x\n");
      },
      "non-direction transcript bytes must fail closed");
}

void replay_validation_uses_exact_codingame_rules() {
  const papersoccer::RulesConfig rules = audit::codingame_rules();
  require(rules.width == 8 && rules.height == 10 &&
              rules.goal_rule == papersoccer::GoalRule::OwnGoalsAllowed &&
              rules.blocked_rule == papersoccer::BlockedRule::MoverLoses,
          "auditor rules must equal production CodinGame rules");
  const auto valid = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "terminal\t0\t0\t0/0/0/0/0/0\n");
  audit::validate_records(valid);
  const auto nonterminal = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "nonterminal\t0\t0\t0\n");
  require_invalid_argument(
      [&] { audit::validate_records(nonterminal); },
      "nonterminal transcripts must be rejected before auditing");
  const auto wrong_winner = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "wrong-winner\t0\t1\t0/0/0/0/0/0\n");
  require_invalid_argument(
      [&] { audit::validate_records(wrong_winner); },
      "winner metadata must agree with native replay");
}

void native_diagnostics_are_complete_and_deterministic() {
  const auto records = read(
      "# agent_id=6600001\n"
      "game_id\tcandidate_player\twinner\tturns\n"
      "deterministic\t0\t0\t0/0/0/0/0/0\n");
  const audit::AuditConfig config = fast_config();
  audit::validate_records(records);
  const auto first = audit::audit_records(records, config);
  const auto second = audit::audit_records(records, config);
  require(first.size() == 3 && second.size() == first.size(),
          "player one must have three own decisions in the fixture");
  for (std::size_t index = 0; index < first.size(); ++index) {
    const auto &left = first[index];
    const auto &right = second[index];
    require(left.state_id == right.state_id &&
                left.search.action == right.search.action &&
                left.classification == right.classification,
            "fixed-work diagnostics must be deterministic");
    require(left.search.stats.tree_nodes <= config.fixed_work &&
                left.root.retained_actions <= config.max_actions &&
                left.search.root_actions > 0,
            "native work and action caps must be reported and enforced");
    require(left.chosen.boundary_retained_ordinal >= 0 &&
                left.chosen.initial_rank > 0 &&
                left.chosen.final_backed_value.has_value() &&
                left.chosen.final_visits.has_value() &&
                left.chosen.final_tactical_class != "not-searched",
            "chosen action must expose retention, initial, and final values");
    if (left.actual.boundary_retained_ordinal >= 0) {
      require(left.actual.initial_action_value.has_value() &&
                  left.actual.initial_rank > 0,
              "retained observed action must expose its initial ordering");
    }
    require(left.root.fifo_extractions + left.root.lifo_extractions ==
                left.root.explored_partial_paths,
            "root deque extraction counters must account for every path");
  }
}

audit::ActionDiagnostic retained(std::string action, int boundary,
                                 std::string tactical, int rank,
                                 bool final_value = true) {
  audit::ActionDiagnostic result;
  result.action = std::move(action);
  result.exact_retained_ordinal = boundary;
  result.boundary_retained_ordinal = boundary;
  result.retained_action = result.action;
  result.tactical_class = std::move(tactical);
  result.initial_action_value = 0.25F;
  result.initial_rank = rank;
  if (final_value) {
    result.final_backed_value = 0.1F;
    result.final_visits = 2;
    result.final_selection_visits = 1;
    result.final_tactical_class = result.tactical_class;
  }
  return result;
}

void classification_taxonomy_is_explicit() {
  audit::SearchDecision search;
  audit::RootDiagnostics root;
  root.initial_best_action = "1";

  auto actual = retained("1", 0, "safe-handoff", 1);
  auto chosen = retained("1", 0, "safe-handoff", 1);
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "match",
          "exact action equality must classify as match");

  chosen.action = "2";
  chosen.retained_action = "1";
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "boundary-equivalent",
          "same boundary must not be called an omission");

  actual = audit::ActionDiagnostic{};
  actual.action = "7";
  chosen = retained("1", 0, "safe-handoff", 1);
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "generator-omission",
          "an absent observed boundary must identify the generator");

  actual = retained("0", 1, "safe-handoff", 2, false);
  search.stats.deadline_reached = true;
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "operational-failure",
          "deadline-root mismatch must identify operational pressure");
  search.stats.deadline_reached = false;

  actual = retained("0", 1, "immediate-goal", 1);
  chosen = retained("2", 2, "safe-handoff", 2);
  root.initial_best_action = "2";
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "tactical-miss",
          "stronger exact observed tactics must take tactical precedence");

  actual = retained("0", 1, "safe-handoff", 1);
  chosen = retained("2", 2, "safe-handoff", 2);
  root.initial_best_action = "0";
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "bfm-override",
          "moving away from the initial leader must identify BFM");

  root.initial_best_action = "2";
  require(audit::classify_decision(actual, chosen, search, root).first ==
              "initial-evaluator-ordering",
          "an unchanged initial leader must identify evaluator ordering");
}

void jsonl_and_tsv_preserve_required_fields() {
  const auto records = read(
      "# agent_id=6600001\n"
      "# run_id=native-audit-test\n"
      "game_id\tcandidate_player\twinner\tturns\n"
      "output\t0\t0\t0/0/0/0/0/0\n");
  audit::AuditConfig config = fast_config();
  audit::validate_records(records);
  const auto rows = audit::audit_records(records, config);

  std::ostringstream jsonl;
  audit::write_audits(jsonl, rows, config);
  const std::string json = jsonl.str();
  require(json.find("\"schema_version\":\"jacek-native-decision-audit-v1\"") !=
              std::string::npos &&
              json.find("\"input_provenance\":{\"agent_id\":\"6600001\"") !=
                  std::string::npos &&
              json.find("\"actual_initial_neural_value\":") !=
                  std::string::npos &&
              json.find("\"chosen_final_backed_value\":") !=
                  std::string::npos &&
              json.find("\"search_generator_truncations\":") !=
                  std::string::npos &&
              json.find("\"search_tactical_proof_paths\":") !=
                  std::string::npos &&
              json.find("\"diagnostic_root_tactical_proof_truncated\":") !=
                  std::string::npos,
          "JSONL must retain provenance and native decision diagnostics");

  config.output_format = audit::OutputFormat::Tsv;
  std::ostringstream tsv;
  audit::write_audits(tsv, rows, config);
  std::istringstream lines(tsv.str());
  std::string header;
  std::string first_row;
  std::getline(lines, header);
  std::getline(lines, first_row);
  require(header == audit::kTsvHeader &&
              field_count(header) == field_count(first_row),
          "TSV header and data rows must have the same strict field count");
}

void clock_and_cli_configuration_are_bounded() {
  const audit::AuditConfig clock = parse_options({"--codingame-clocks"});
  require(clock.mode == audit::AuditMode::Clock &&
              clock.first_time_ms == 800 && clock.later_time_ms == 155 &&
              clock.fixed_work ==
                  papersoccer::jacek_native_bfm::kProductionTreeNodes,
          "CodinGame mode must freeze the production 800/155 configuration");
  require(audit::decision_time_limit(clock, 0) == 800 &&
              audit::decision_time_limit(clock, 1) == 155,
          "first clock must be keyed to first own decision");
  const audit::AuditConfig fixed =
      parse_options({"--fixed-work", "251", "--format", "tsv"});
  require(fixed.mode == audit::AuditMode::FixedWork &&
              fixed.fixed_work == 251 &&
              fixed.output_format == audit::OutputFormat::Tsv,
          "explicit fixed-work mode must remain deterministic");
  require_invalid_argument(
      [] { (void)parse_options({"--fixed-work", "1"}); },
      "fixed work below two nodes must be rejected");
  require_invalid_argument(
      [] { (void)parse_options({"--first-ms", "800"}); },
      "custom clock mode requires both first and later limits");
  require_invalid_argument(
      [] {
        (void)parse_options(
            {"--codingame-clocks", "--first-ms", "900", "--later-ms", "180"});
      },
      "canonical and custom clock modes must not be conflated");
}

void run_validates_all_input_before_writing() {
  std::istringstream input(
      "game_id\tcandidate_player\twinner\tturns\n"
      "nonterminal\t0\t0\t0\n");
  std::ostringstream output;
  std::ostringstream error;
  std::vector<std::string> options{"auditor", "--fixed-work", "64",
                                   "--max-actions", "16",
                                   "--max-partial-paths", "64"};
  std::vector<char *> arguments;
  for (std::string &option : options) arguments.push_back(option.data());
  const int status = audit::run(static_cast<int>(arguments.size()),
                                arguments.data(), input, output, error);
  require(status == 2 && output.str().empty() &&
              error.str().find("not terminal") != std::string::npos,
          "invalid replay input must fail before emitting any partial audit");
}

}  // namespace

int main() {
  try {
    strict_collector_contract_and_provenance();
    replay_validation_uses_exact_codingame_rules();
    native_diagnostics_are_complete_and_deterministic();
    classification_taxonomy_is_explicit();
    jsonl_and_tsv_preserve_required_fields();
    clock_and_cli_configuration_are_bounded();
    run_validates_all_input_before_writing();
    std::cout << "Jacek-native replay decision auditor tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
