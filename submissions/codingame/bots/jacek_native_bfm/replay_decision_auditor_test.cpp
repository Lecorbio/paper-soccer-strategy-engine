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

void selected_prefix_manifest_is_exact_and_fail_closed() {
  const auto records = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "selected\t0\t0\t0/0/0/0/0/0\n");
  audit::validate_records(records);
  papersoccer::GameState initial =
      papersoccer::make_initial_state(audit::codingame_rules());
  const std::string state_id = audit::state_identity(initial);
  std::istringstream manifest(
      "game_id\tturn_index\tstate_id\nselected\t0\t" + state_id + "\n");
  auto selected = audit::read_selected_prefixes(manifest);
  const auto rows = audit::audit_records(records, fast_config(), &selected);
  require(rows.size() == 1 && rows.front().turn_index == 0 &&
              selected.front().matched,
          "an exact manifest must search only the named candidate decision");

  std::istringstream duplicate(
      "game_id\tturn_index\tstate_id\nselected\t0\t" + state_id +
      "\nselected\t0\t" + state_id + "\n");
  require_invalid_argument(
      [&] { (void)audit::read_selected_prefixes(duplicate); },
      "duplicate manifest keys must fail closed");
  std::istringstream wrong(
      "game_id\tturn_index\tstate_id\nselected\t0\tfnv1a64:0000000000000000\n");
  auto wrong_selection = audit::read_selected_prefixes(wrong);
  require_invalid_argument(
      [&] { (void)audit::audit_records(records, fast_config(),
                                       &wrong_selection); },
      "a stale manifest state identity must fail closed");
  std::istringstream unknown(
      "game_id\tturn_index\tstate_id\nmissing\t0\t" + state_id + "\n");
  auto unknown_selection = audit::read_selected_prefixes(unknown);
  require_invalid_argument(
      [&] { (void)audit::audit_records(records, fast_config(),
                                       &unknown_selection); },
      "an unknown manifest decision must fail closed");
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
                left.search.root_actions > 0 &&
                left.search.root_action_diagnostics.size() ==
                    left.search.root_actions,
            "native work and action caps must be reported and enforced");
    require(left.runtime.runtime_sha256 ==
                "17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3" &&
                left.runtime.model_sha256 ==
                    papersoccer::jacek_native_model::kModelSha256 &&
                left.runtime.packed_sha256 ==
                    papersoccer::jacek_native_model::kPackedSha256,
            "every decision must bind the exact compile-time runtime identity");
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

void strict_runtime_loader_binds_and_rejects_payload_tamper() {
  const std::string raw = audit::default_runtime_text();
  const audit::LoadedRuntime runtime = audit::parse_runtime(raw);
  require(runtime.identity.runtime_sha256 ==
              "17038c104bf79c4d5c4c47f09ea144acdeb5dc8e2b01137d46f6b0c589d304c3" &&
              runtime.identity.model_sha256 ==
                  papersoccer::jacek_native_model::kModelSha256 &&
              runtime.identity.packed_sha256 ==
                  papersoccer::jacek_native_model::kPackedSha256,
          "the strict loader must reproduce the frozen history-62 runtime");
  std::string tampered = raw;
  const std::size_t payload = tampered.rfind('\n', tampered.size() - 2U);
  require(payload != std::string::npos && payload + 1U < tampered.size() - 1U,
          "the runtime fixture must expose one packed payload line");
  tampered[payload + 1U] = tampered[payload + 1U] == 'A' ? 'B' : 'A';
  require_invalid_argument(
      [&] { (void)audit::parse_runtime(tampered); },
      "packed runtime tamper must fail its declared SHA-256");
  require_invalid_argument(
      [&] { (void)audit::parse_runtime(raw.substr(0, raw.size() - 1U)); },
      "runtime checkpoints without their canonical final LF must fail closed");

  std::string alternate = raw;
  const std::string original_model =
      std::string(papersoccer::jacek_native_model::kModelSha256);
  const std::string alternate_model(64, '2');
  const std::size_t model_offset = alternate.find(original_model);
  require(model_offset != std::string::npos,
          "the runtime fixture must contain its model identity");
  alternate.replace(model_offset, original_model.size(), alternate_model);
  const audit::LoadedRuntime loaded = audit::parse_runtime(alternate);
  const auto records = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "alternate\t0\t0\t0/0/0/0/0/0\n");
  audit::validate_records(records);
  const auto rows =
      audit::audit_records(records, fast_config(), nullptr, &loaded);
  require(!rows.empty() && rows.front().runtime.model_sha256 == alternate_model &&
              rows.front().runtime.runtime_sha256 ==
                  loaded.identity.runtime_sha256,
          "auditor rows must bind the loaded runtime, never default_model");
}

void decision_limit_preserves_full_replay_and_first_decision_semantics() {
  const auto records = read(
      "game_id\tcandidate_player\twinner\tturns\n"
      "first-only\t0\t0\t0/0/0/0/0/0\n");
  audit::AuditConfig unlimited = fast_config();
  audit::validate_records(records);
  require(audit::audit_records(records, unlimited).size() == 3,
          "the default must continue to audit every candidate decision");

  audit::AuditConfig first_only = fast_config();
  first_only.max_own_decisions_per_game = 1;
  const auto limited = audit::audit_records(records, first_only);
  require(limited.size() == 1 && limited.front().own_decision_index == 0 &&
              limited.front().turn_index == 0 &&
              limited.front().time_limit_ms == 0,
          "a one-decision limit must audit only candidate decision zero");

  std::istringstream invalid_tail(
      "game_id\tcandidate_player\twinner\tturns\n"
      "turn-after-terminal\t0\t0\t0/0/0/0/0/0/0\n");
  std::ostringstream output;
  std::ostringstream error;
  std::vector<std::string> options{
      "auditor", "--fixed-work", "64", "--max-actions", "16",
      "--max-partial-paths", "64", "--max-own-decisions-per-game", "1"};
  std::vector<char *> arguments;
  for (std::string &option : options) arguments.push_back(option.data());
  const int status = audit::run(static_cast<int>(arguments.size()),
                                arguments.data(), invalid_tail, output, error);
  require(status == 2 && output.str().empty() &&
              error.str().find("after terminal") != std::string::npos,
          "decision filtering must not skip full transcript validation");
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
  config.max_own_decisions_per_game = 1;
  audit::validate_records(records);
  const auto rows = audit::audit_records(records, config);
  require(rows.size() == 1,
          "the bound output configuration must match the filtered rows");

  std::ostringstream jsonl;
  audit::write_audits(jsonl, rows, config);
  const std::string json = jsonl.str();
  require(json.find("\"schema_version\":\"jacek-native-decision-audit-v4\"") !=
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
              json.find("\"root_reply_width\":0") != std::string::npos &&
              json.find("\"supported_advance_penalty\":0") !=
                  std::string::npos &&
              json.find("\"supported_advance_start_progress\":2") !=
                  std::string::npos &&
              json.find("\"supported_advance_endpoint_progress_threshold\":4") !=
                  std::string::npos &&
              json.find("\"maximum_supported_advance_penalty\":0.2") !=
                  std::string::npos &&
              json.find("\"pre_action_used_edges\":0") !=
                  std::string::npos &&
              json.find("\"runtime_sha256\":\"17038c104") !=
                  std::string::npos &&
              json.find("\"actual_exact_reply_refuted\":false") !=
                  std::string::npos &&
              json.find("\"chosen_supported_advance_eligible\":") !=
                  std::string::npos &&
              json.find("\"search_root_reply_probe_candidates\":0") !=
                  std::string::npos &&
              json.find("\"search_exact_reply_refuted_actions\":[]") !=
                  std::string::npos &&
              json.find("\"search_root_action_diagnostics\":[{") !=
                  std::string::npos &&
              json.find("\"initial_value\":") != std::string::npos &&
              json.find("\"diagnostic_root_tactical_proof_truncated\":") !=
                  std::string::npos &&
              json.find("\"max_own_decisions_per_game\":1") !=
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
                  papersoccer::jacek_native_bfm::kProductionTreeNodes &&
              !clock.max_own_decisions_per_game.has_value(),
          "CodinGame mode must freeze the production 800/155 configuration");
  const audit::AuditConfig first_only = parse_options(
      {"--codingame-clocks", "--max-own-decisions-per-game", "1"});
  require(first_only.max_own_decisions_per_game == 1,
          "the bounded first-decision panel option must compose with clocks");
  require(audit::decision_time_limit(clock, 0) == 800 &&
              audit::decision_time_limit(clock, 1) == 155,
          "first clock must be keyed to first own decision");
  const audit::AuditConfig larger_clock_tree =
      parse_options({"--codingame-clocks", "--tree-nodes", "120000",
                     "--root-reply-width", "32",
                     "--supported-advance-penalty", "0.15",
                     "--checkpoint", "seed.runtime"});
  require(larger_clock_tree.mode == audit::AuditMode::Clock &&
              larger_clock_tree.fixed_work == 120000 &&
              larger_clock_tree.root_reply_width == 32 &&
              larger_clock_tree.supported_advance_penalty == 0.15 &&
              larger_clock_tree.checkpoint_path == "seed.runtime" &&
              larger_clock_tree.first_time_ms == 800 &&
              larger_clock_tree.later_time_ms == 155,
          "clock mode must honor an explicit supported tree-node cap");
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
  require_invalid_argument(
      [] {
        (void)parse_options(
            {"--codingame-clocks", "--fixed-work", "120000"});
      },
      "clock mode must use the explicit tree-node option, not fixed work");
  require_invalid_argument(
      [] {
        (void)parse_options(
            {"--codingame-clocks", "--tree-nodes", "120001"});
      },
      "clock-mode tree caps above the native maximum must fail closed");
  require_invalid_argument(
      [] { (void)parse_options({"--root-reply-width", "251"}); },
      "reply width above the root action limit must fail closed");
  require_invalid_argument(
      [] {
        (void)parse_options({"--supported-advance-penalty", "-0.01"});
      },
      "negative supported-advance penalties must fail closed");
  require_invalid_argument(
      [] {
        (void)parse_options({"--supported-advance-penalty", "0.201"});
      },
      "supported-advance penalties above the native bound must fail closed");
  require_invalid_argument(
      [] { (void)parse_options({"--max-own-decisions-per-game", "0"}); },
      "a zero decision limit must fail closed");
  require_invalid_argument(
      [] { (void)parse_options({"--max-own-decisions-per-game", "1025"}); },
      "a decision limit above the transcript bound must fail closed");
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

  std::istringstream valid_input(
      "game_id\tcandidate_player\twinner\tturns\n"
      "terminal\t0\t0\t0/0/0/0/0/0\n");
  std::ostringstream missing_output;
  std::ostringstream missing_error;
  std::vector<std::string> missing_options{
      "auditor", "--checkpoint", "/definitely/missing/jacek.runtime"};
  std::vector<char *> missing_arguments;
  for (std::string &option : missing_options) {
    missing_arguments.push_back(option.data());
  }
  const int missing_status = audit::run(
      static_cast<int>(missing_arguments.size()), missing_arguments.data(),
      valid_input, missing_output, missing_error);
  require(missing_status == 2 && missing_output.str().empty() &&
              missing_error.str().find("cannot open runtime checkpoint") !=
                  std::string::npos,
          "an unavailable external runtime must fail before any audit output");
}

}  // namespace

int main() {
  try {
    strict_collector_contract_and_provenance();
    replay_validation_uses_exact_codingame_rules();
    selected_prefix_manifest_is_exact_and_fail_closed();
    native_diagnostics_are_complete_and_deterministic();
    strict_runtime_loader_binds_and_rejects_payload_tamper();
    decision_limit_preserves_full_replay_and_first_decision_semantics();
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
