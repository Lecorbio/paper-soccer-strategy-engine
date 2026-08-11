#define PAPER_SOCCER_JACEK_NATIVE_RESTART_NO_MAIN
#include "../../tools/jacek_native_restart_round2.cpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {

[[maybe_unused]] const auto kSelectReanalysisReference =
    &select_reanalysis_indices;
[[maybe_unused]] const auto kClassifyReanalysisReference =
    &classify_reanalysis;
[[maybe_unused]] const auto kParseRestartReference =
    &parse_restart_arguments;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

template <typename Function>
void require_invalid(Function &&function, std::string_view message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

std::string metadata(std::string_view extra = {}) {
  return std::string(
             "# agent_id=6609056\n"
             "# arena_manifest_sha256=") + std::string(64, 'a') +
         "\n# asserted_source_sha256=" + std::string(64, 'b') +
         "\n# asserted_submission_id=41123817"
         "\n# collector_sha256=" + std::string(64, 'c') +
         "\n# exclusion_registry_sha256=" + std::string(64, 'd') +
         "\n# repository_commit=" + std::string(40, 'e') +
         "\n# run_id=restart-test"
         "\n# source_binding_status=asserted-not-api-verified\n" +
         std::string(extra);
}

std::filesystem::path temporary_path(std::string_view suffix) {
  return std::filesystem::temp_directory_path() /
         ("papersoccer-restart-" + std::to_string(
              std::chrono::steady_clock::now().time_since_epoch().count()) +
          std::string(suffix));
}

CollectorInput read_text(std::string_view value) {
  const std::filesystem::path path = temporary_path(".tsv");
  {
    std::ofstream output(path, std::ios::binary);
    output << value;
  }
  try {
    CollectorInput result = read_collector(path.string());
    std::filesystem::remove(path);
    return result;
  } catch (...) {
    std::filesystem::remove(path);
    throw;
  }
}

std::string valid_loss_tsv() {
  return metadata() +
         "game_id\tcandidate_player\twinner\tturns\n"
         "1001\t1\t0\t0/0/0/0/0/0\n";
}

void exact_collector_schema_and_identity() {
  const CollectorInput input = read_text(valid_loss_tsv());
  require(input.records.size() == 1U && input.metadata.size() == 9U,
          "the exact collector schema should parse");
  require(input.sha256 == native::sha256_hex(std::span<const std::uint8_t>(
              reinterpret_cast<const std::uint8_t *>(input.bytes.data()),
              input.bytes.size())),
          "the producer must hash the exact collector bytes");
  require_invalid(
      [] {
        (void)read_text(metadata("# unknown=value\n") +
            "game_id\tcandidate_player\twinner\tturns\n"
            "1001\t1\t0\t0/0/0/0/0/0\n");
      },
      "unknown metadata must fail the exact schema");
  require_invalid(
      [] {
        (void)read_text(metadata() +
            "game_id\tcandidate_player\twinner\tturns\n"
            "1001\t1\t0\t0/0/0/0/0/0\n"
            "1001\t1\t0\t0/0/0/0/0/0\n");
      },
      "duplicate games must fail closed");
}

void full_replay_and_loss_filtering() {
  const CollectorInput input = read_text(valid_loss_tsv());
  const std::vector<RestartPrefix> first =
      validate_and_select_prefixes(input, 2U);
  const std::vector<RestartPrefix> second =
      validate_and_select_prefixes(input, 2U);
  require(first.size() == 2U && first[0].prefix_turn == 1U &&
              first[1].prefix_turn == 5U,
          "candidate decision prefixes must be selected evenly");
  require(first[0].state_id == second[0].state_id &&
              first[1].transcript == "0/0/0/0/0" &&
              first[0].candidate_player == 1 && first[0].observed_winner == 0,
          "prefix selection must be deterministic and loss-bound");
  const std::vector<RestartPrefix> limited =
      limit_restart_prefixes(first, 1U);
  require(limited.size() == 1U && limited[0].prefix_turn == 5U,
          "the fixed pilot cap must select the deterministic middle prefix");
  require(first[0].state.config.goal_rule == ps::GoalRule::OwnGoalsAllowed &&
              first[0].state.config.blocked_rule == ps::BlockedRule::MoverLoses,
          "replay must use the exact CodinGame rules");
  require_invalid(
      [] {
        const CollectorInput nonterminal = read_text(
            metadata() + "game_id\tcandidate_player\twinner\tturns\n"
                         "1002\t1\t0\t0\n");
        (void)validate_and_select_prefixes(nonterminal, 2U);
      },
      "operationally incomplete/nonterminal games must be rejected");
  require_invalid(
      [] {
        const CollectorInput malformed = read_text(
            metadata() + "game_id\tcandidate_player\twinner\tturns\n"
                         "1003\t1\t0\t000000\n");
        (void)validate_and_select_prefixes(malformed, 2U);
      },
      "a complete turn that continues after handoff must be rejected");
  require_invalid(
      [] {
        const CollectorInput win_only = read_text(
            metadata() + "game_id\tcandidate_player\twinner\tturns\n"
                         "1004\t0\t0\t0/0/0/0/0/0\n");
        (void)validate_and_select_prefixes(win_only, 2U);
      },
      "candidate wins must never become live-loss restarts");
}

void explicit_provenance_and_no_observed_labels() {
  const CollectorInput input = read_text(valid_loss_tsv());
  const RestartPrefix prefix = validate_and_select_prefixes(input, 1U).front();
  RestartArguments arguments;
  arguments.work = 64;
  arguments.temperature = 0.0F;
  arguments.temperature_turns = 0;
  arguments.producer_sha256 = std::string(64, '1');
  arguments.build_provenance_sha256 = std::string(64, '2');
  LoadedModel player_one;
  LoadedModel player_two;
  std::ostringstream output;
  write_restart_record(output, arguments, input, prefix, 0U, 0U, 7U, 0,
                       prefix.prefix_turn + 1U, {}, player_one, player_two,
                       nullptr, prefix.transcript + "/0", {});
  const std::string rendered = output.str();
  require(rendered.find("\"observed_moves_usage\":\"state-construction-only\"")
              != std::string::npos &&
              rendered.find("\"policy_target\":null") != std::string::npos,
          "observed moves must be provenance-only state construction");
  require(rendered.find("teacher_move") == std::string::npos &&
              rendered.find("actual_action") == std::string::npos &&
              rendered.find(metadata_value(input, "arena_manifest_sha256"))
                  != std::string::npos,
          "records must carry arena provenance without action labels");

  RestartArguments expected;
  expected.input_sha256 = input.sha256;
  expected.expected_source_sha256 =
      metadata_value(input, "asserted_source_sha256");
  expected.expected_manifest_sha256 =
      metadata_value(input, "arena_manifest_sha256");
  expected.expected_exclusion_registry_sha256 =
      metadata_value(input, "exclusion_registry_sha256");
  expected.expected_submission_id =
      metadata_value(input, "asserted_submission_id");
  expected.expected_agent_id = metadata_value(input, "agent_id");
  assert_expected_collector(expected, input);
  expected.expected_agent_id = "1";
  require_invalid(
      [&] { assert_expected_collector(expected, input); },
      "an explicit expected identity mismatch must fail closed");
}

}  // namespace

int main() {
  try {
    exact_collector_schema_and_identity();
    full_replay_and_loss_filtering();
    explicit_provenance_and_no_observed_labels();
    std::cout << "Jacek-native live restart tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
