#include <charconv>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <span>
#include <unordered_set>

// This is an additive producer.  The ordinary round-two league stays frozen;
// only its native search, feature, checkpoint, and reanalysis primitives are
// reused here.
#define PAPER_SOCCER_JACEK_NATIVE_ROUND2_NO_MAIN
#include "jacek_native_selfplay_round2.cpp"

namespace {

constexpr std::string_view kRestartGameSchema =
    "papersoccer.jacek-native-restart-game/v1";
constexpr std::string_view kRestartGeneratorSchema =
    "jacek-native-live-restart-bfm/v1";
constexpr std::string_view kRestartOpeningSchema =
    "collector-clean-candidate-loss-prefix/v1";
constexpr std::string_view kRestartColorSchedule =
    "swap-player-checkpoints-on-odd-continuations/v1";
constexpr std::string_view kRestartTemperatureSchedule =
    "restart-relative-complete-turn-index-before-cutoff/v1";
constexpr std::string_view kCollectorHeader =
    "game_id\tcandidate_player\twinner\tturns";
constexpr std::size_t kMaximumCollectorBytes = 32U * 1024U * 1024U;
constexpr std::size_t kMaximumCollectorGames = 512U;
constexpr std::size_t kMaximumCollectorTurns = 1'024U;
constexpr std::size_t kMaximumCollectorActionBytes = 65'536U;

// Including the round-two implementation with its CLI disabled leaves this
// ordinary-league scheduler intentionally unused in the restart producer.
[[maybe_unused]] const auto kRound2SchedulerReference =
    &schedule_round2_game;

const std::array<std::string_view, 9> kRequiredMetadata{{
    "agent_id",
    "arena_manifest_sha256",
    "asserted_source_sha256",
    "asserted_submission_id",
    "collector_sha256",
    "exclusion_registry_sha256",
    "repository_commit",
    "run_id",
    "source_binding_status",
}};

struct RestartArguments {
  std::string input;
  std::string output;
  std::string input_sha256;
  std::string expected_source_sha256;
  std::string expected_manifest_sha256;
  std::string expected_exclusion_registry_sha256;
  std::string expected_submission_id;
  std::string expected_agent_id;
  std::string player_one_checkpoint;
  std::string player_two_checkpoint;
  std::string player_one_artifact_sha256;
  std::string player_two_artifact_sha256;
  std::string reanalysis_checkpoint;
  std::string reanalysis_artifact_sha256;
  std::string producer_sha256;
  std::string build_provenance_sha256;
  std::uint64_t seed{20260831ULL};
  std::uint64_t work{4'096ULL};
  std::uint64_t reanalysis_work{};
  std::uint64_t verification_work{kRound2VerificationWork};
  std::size_t samples_per_game{100U};
  std::size_t reanalysis_samples_per_game{12U};
  std::size_t prefixes_per_loss{4U};
  std::size_t maximum_selected_prefixes{};
  std::size_t continuations_per_prefix{2U};
  std::size_t shard_index{};
  std::size_t shard_count{1U};
  float temperature{3.0F};
  std::size_t temperature_turns{12U};
  std::size_t maximum_generated_turns{384U};
};

struct CollectorRecord {
  std::string game_id;
  int candidate_player{};
  int winner{};
  std::vector<std::string> actions;
};

struct CollectorInput {
  std::string bytes;
  std::string sha256;
  std::map<std::string, std::string> metadata;
  std::vector<CollectorRecord> records;
};

struct RestartPrefix {
  std::string game_id;
  int candidate_player{};
  int observed_winner{};
  std::size_t observed_turns{};
  std::size_t prefix_turn{};
  std::string transcript;
  std::string state_id;
  ps::GameState state;
};

std::vector<std::string_view> split_exact(std::string_view text,
                                          char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  while (true) {
    const std::size_t end = text.find(separator, begin);
    result.push_back(text.substr(
        begin, end == std::string_view::npos ? text.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return result;
    begin = end + 1U;
  }
}

template <typename Integer>
Integer parse_decimal_exact(std::string_view text, std::string_view label) {
  Integer value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(std::string(label) +
                                " must be an unsigned decimal integer");
  }
  return value;
}

bool safe_identifier(std::string_view value, std::size_t maximum = 128U) {
  return !value.empty() && value.size() <= maximum &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') ||
                  character == '-' || character == '_' || character == '.' ||
                  character == ':';
         });
}

bool lower_hex(std::string_view value, std::size_t length) {
  return value.size() == length &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::string metadata_value(const CollectorInput &input,
                           std::string_view key) {
  const auto found = input.metadata.find(std::string(key));
  if (found == input.metadata.end()) {
    throw std::invalid_argument("collector metadata is missing " +
                                std::string(key));
  }
  return found->second;
}

void validate_collector_metadata(const CollectorInput &input) {
  if (input.metadata.size() != kRequiredMetadata.size()) {
    throw std::invalid_argument(
        "collector TSV metadata fields do not match the frozen schema");
  }
  for (const std::string_view key : kRequiredMetadata) {
    if (!input.metadata.contains(std::string(key))) {
      throw std::invalid_argument("collector TSV metadata is missing " +
                                  std::string(key));
    }
  }
  for (const std::string_view key : {
           std::string_view{"arena_manifest_sha256"},
           std::string_view{"asserted_source_sha256"},
           std::string_view{"collector_sha256"},
           std::string_view{"exclusion_registry_sha256"}}) {
    if (!lower_hex(metadata_value(input, key), 64U)) {
      throw std::invalid_argument("collector TSV has invalid " +
                                  std::string(key));
    }
  }
  (void)parse_decimal_exact<std::uint64_t>(
      metadata_value(input, "agent_id"), "agent_id");
  (void)parse_decimal_exact<std::uint64_t>(
      metadata_value(input, "asserted_submission_id"),
      "asserted_submission_id");
  const std::string commit = metadata_value(input, "repository_commit");
  if (commit.size() != 40U || !lower_hex(commit, commit.size())) {
    throw std::invalid_argument("collector TSV repository_commit is not exact");
  }
  if (!safe_identifier(metadata_value(input, "run_id"))) {
    throw std::invalid_argument("collector TSV run_id is unsafe");
  }
  const std::string binding =
      metadata_value(input, "source_binding_status");
  if (binding != "asserted-not-api-verified" && binding != "api-verified") {
    throw std::invalid_argument(
        "collector TSV source_binding_status is unsupported");
  }
}

CollectorInput read_collector(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("could not open explicit collector TSV");
  CollectorInput result;
  result.bytes.assign(std::istreambuf_iterator<char>(input),
                      std::istreambuf_iterator<char>());
  if (result.bytes.empty() || result.bytes.size() > kMaximumCollectorBytes ||
      result.bytes.find('\0') != std::string::npos ||
      result.bytes.find('\r') != std::string::npos) {
    throw std::invalid_argument("collector TSV bytes are not canonical LF text");
  }
  result.sha256 = native::sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(result.bytes.data()),
      result.bytes.size()));

  bool saw_header = false;
  std::unordered_set<std::string> game_ids;
  const auto lines = split_exact(result.bytes, '\n');
  for (std::size_t index = 0; index < lines.size(); ++index) {
    const std::string_view line = lines[index];
    if (line.empty()) {
      if (index + 1U == lines.size()) continue;
      throw std::invalid_argument("collector TSV contains a blank line");
    }
    if (!saw_header && line.starts_with("# ")) {
      const std::size_t equals = line.find('=');
      if (equals <= 2U || equals == std::string_view::npos ||
          line.find('=', equals + 1U) != std::string_view::npos) {
        throw std::invalid_argument("collector TSV metadata syntax is invalid");
      }
      const std::string key(line.substr(2U, equals - 2U));
      const std::string value(line.substr(equals + 1U));
      if (!safe_identifier(key, 64U) || value.empty() || value.size() > 512U ||
          !std::all_of(value.begin(), value.end(), [](char character) {
            return character >= 0x20 && character <= 0x7e;
          }) ||
          !result.metadata.emplace(key, value).second) {
        throw std::invalid_argument(
            "collector TSV metadata is unsafe or duplicated");
      }
      continue;
    }
    if (!saw_header) {
      if (line != kCollectorHeader) {
        throw std::invalid_argument("collector TSV header is not exact");
      }
      saw_header = true;
      continue;
    }
    if (line.front() == '#') {
      throw std::invalid_argument("collector metadata appears after the header");
    }
    if (result.records.size() >= kMaximumCollectorGames) {
      throw std::invalid_argument("collector TSV exceeds the game limit");
    }
    const auto fields = split_exact(line, '\t');
    if (fields.size() != 4U || fields[0].empty() || fields[0].size() > 32U ||
        !std::all_of(fields[0].begin(), fields[0].end(), [](char character) {
          return character >= '0' && character <= '9';
        })) {
      throw std::invalid_argument("collector TSV game row schema is invalid");
    }
    (void)parse_decimal_exact<std::uint64_t>(fields[0], "game_id");
    if (!game_ids.insert(std::string(fields[0])).second) {
      throw std::invalid_argument("collector TSV contains duplicate game_id");
    }
    if (fields[1].size() != 1U || fields[2].size() != 1U ||
        (fields[1] != "0" && fields[1] != "1") ||
        (fields[2] != "0" && fields[2] != "1") || fields[3].empty()) {
      throw std::invalid_argument("collector TSV player fields are invalid");
    }
    CollectorRecord record;
    record.game_id = fields[0];
    record.candidate_player = fields[1][0] - '0';
    record.winner = fields[2][0] - '0';
    const auto actions = split_exact(fields[3], '/');
    if (actions.empty() || actions.size() > kMaximumCollectorTurns) {
      throw std::invalid_argument("collector TSV turn count is invalid");
    }
    record.actions.reserve(actions.size());
    for (const std::string_view action : actions) {
      if (action.empty() || action.size() > kMaximumCollectorActionBytes ||
          !std::all_of(action.begin(), action.end(), [](char direction) {
            return direction >= '0' && direction <= '7';
          })) {
        throw std::invalid_argument(
            "collector TSV contains an invalid complete-turn action");
      }
      record.actions.emplace_back(action);
    }
    result.records.push_back(std::move(record));
  }
  if (!saw_header || result.records.empty()) {
    throw std::invalid_argument("collector TSV has no complete game rows");
  }
  validate_collector_metadata(result);
  return result;
}

ps::RulesConfig restart_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

std::vector<RestartPrefix> validate_and_select_prefixes(
    const CollectorInput &input, std::size_t prefixes_per_loss) {
  std::vector<RestartPrefix> selected;
  std::unordered_set<std::string> selected_states;
  std::size_t losses = 0;
  for (const CollectorRecord &record : input.records) {
    ps::GameState state = ps::make_initial_state(restart_rules());
    std::string transcript;
    std::vector<RestartPrefix> eligible;
    if (record.winner != record.candidate_player) ++losses;
    for (std::size_t turn = 0; turn < record.actions.size(); ++turn) {
      if (ps::is_terminal(state)) {
        throw std::invalid_argument("collector game " + record.game_id +
                                    " continues after terminal state");
      }
      if (record.winner != record.candidate_player && turn != 0U &&
          (state.to_move == ps::Player::One ? 0 : 1) ==
              record.candidate_player) {
        const std::string state_id = feature_id(active_features(state));
        eligible.push_back(RestartPrefix{
            record.game_id, record.candidate_player, record.winner,
            record.actions.size(), turn, transcript, state_id, state});
      }
      try {
        native::apply_encoded_turn(state, record.actions[turn]);
      } catch (const std::exception &error) {
        throw std::invalid_argument("collector game " + record.game_id +
                                    " has invalid complete turn " +
                                    std::to_string(turn) + ": " + error.what());
      }
      if (!transcript.empty()) transcript.push_back('/');
      transcript += record.actions[turn];
    }
    if (!ps::is_terminal(state) || !ps::winner(state).has_value() ||
        (*ps::winner(state) == ps::Player::One ? 0 : 1) != record.winner) {
      throw std::invalid_argument("collector game " + record.game_id +
                                  " is nonterminal or has mismatched winner");
    }
    const std::size_t count =
        std::min(prefixes_per_loss, eligible.size());
    for (std::size_t index = 0; index < count; ++index) {
      const std::size_t source =
          count == 1U ? 0U : index * (eligible.size() - 1U) / (count - 1U);
      RestartPrefix prefix = std::move(eligible[source]);
      if (selected_states.insert(prefix.state_id).second) {
        selected.push_back(std::move(prefix));
      }
    }
  }
  if (losses == 0U) {
    throw std::invalid_argument("collector TSV contains no candidate losses");
  }
  if (selected.empty()) {
    throw std::invalid_argument(
        "candidate losses contain no noninitial candidate decision prefixes");
  }
  return selected;
}

std::vector<RestartPrefix> limit_restart_prefixes(
    std::vector<RestartPrefix> prefixes, std::size_t maximum) {
  if (maximum == 0U || prefixes.size() <= maximum) return prefixes;
  std::vector<RestartPrefix> limited;
  limited.reserve(maximum);
  if (maximum == 1U) {
    limited.push_back(std::move(prefixes[prefixes.size() / 2U]));
    return limited;
  }
  for (std::size_t index = 0; index < maximum; ++index) {
    const std::size_t source =
        index * (prefixes.size() - 1U) / (maximum - 1U);
    limited.push_back(std::move(prefixes[source]));
  }
  return limited;
}

RestartArguments parse_restart_arguments(int argc, char **argv) {
  RestartArguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("every restart option requires a value");
    }
    const std::string_view value = argv[++index];
    if (option == "--input") result.input = value;
    else if (option == "--output") result.output = value;
    else if (option == "--input-sha256") result.input_sha256 = value;
    else if (option == "--expected-source-sha256") {
      result.expected_source_sha256 = value;
    } else if (option == "--expected-manifest-sha256") {
      result.expected_manifest_sha256 = value;
    } else if (option == "--expected-exclusion-registry-sha256") {
      result.expected_exclusion_registry_sha256 = value;
    } else if (option == "--expected-submission-id") {
      result.expected_submission_id = value;
    } else if (option == "--expected-agent-id") {
      result.expected_agent_id = value;
    } else if (option == "--player-one-checkpoint") {
      result.player_one_checkpoint = value;
    } else if (option == "--player-two-checkpoint") {
      result.player_two_checkpoint = value;
    } else if (option == "--player-one-artifact-sha256") {
      result.player_one_artifact_sha256 = value;
    } else if (option == "--player-two-artifact-sha256") {
      result.player_two_artifact_sha256 = value;
    } else if (option == "--reanalysis-checkpoint") {
      result.reanalysis_checkpoint = value;
    } else if (option == "--reanalysis-artifact-sha256") {
      result.reanalysis_artifact_sha256 = value;
    } else if (option == "--producer-sha256") {
      result.producer_sha256 = value;
    } else if (option == "--build-provenance-sha256") {
      result.build_provenance_sha256 = value;
    } else if (option == "--seed") {
      result.seed = parse_round2_unsigned(value, "seed");
    } else if (option == "--work") {
      result.work = parse_round2_unsigned(value, "work");
    } else if (option == "--reanalysis-work") {
      result.reanalysis_work =
          parse_round2_unsigned(value, "reanalysis work");
    } else if (option == "--verification-work") {
      result.verification_work =
          parse_round2_unsigned(value, "verification work");
    } else if (option == "--samples-per-game") {
      result.samples_per_game =
          parse_round2_unsigned(value, "samples per game");
    } else if (option == "--reanalysis-samples-per-game") {
      result.reanalysis_samples_per_game =
          parse_round2_unsigned(value, "reanalysis samples per game");
    } else if (option == "--prefixes-per-loss") {
      result.prefixes_per_loss =
          parse_round2_unsigned(value, "prefixes per loss");
    } else if (option == "--max-selected-prefixes") {
      result.maximum_selected_prefixes =
          parse_round2_unsigned(value, "maximum selected prefixes");
    } else if (option == "--continuations-per-prefix") {
      result.continuations_per_prefix =
          parse_round2_unsigned(value, "continuations per prefix");
    } else if (option == "--shard-index") {
      result.shard_index = parse_round2_unsigned(value, "shard index");
    } else if (option == "--shard-count") {
      result.shard_count = parse_round2_unsigned(value, "shard count");
    } else if (option == "--temperature") {
      result.temperature = parse_float(value, "temperature");
    } else if (option == "--temperature-turns") {
      result.temperature_turns =
          parse_round2_unsigned(value, "temperature turns");
    } else if (option == "--max-generated-complete-turns") {
      result.maximum_generated_turns =
          parse_round2_unsigned(value, "maximum generated turns");
    } else {
      throw std::invalid_argument("unknown restart option " +
                                  std::string(option));
    }
  }
  const bool reanalysis_enabled = result.reanalysis_work != 0U;
  if (result.input.empty() || result.output.empty() ||
      !valid_sha256(result.input_sha256) ||
      !valid_sha256(result.expected_source_sha256) ||
      !valid_sha256(result.expected_manifest_sha256) ||
      !valid_sha256(result.expected_exclusion_registry_sha256) ||
      result.expected_submission_id.empty() || result.expected_agent_id.empty() ||
      result.player_one_checkpoint.empty() ||
      result.player_two_checkpoint.empty() ||
      !valid_sha256(result.player_one_artifact_sha256) ||
      !valid_sha256(result.player_two_artifact_sha256) ||
      !valid_sha256(result.producer_sha256) ||
      !valid_sha256(result.build_provenance_sha256) || result.work < 2U ||
      result.samples_per_game == 0U || result.samples_per_game > 100U ||
      result.reanalysis_samples_per_game > result.samples_per_game ||
      result.prefixes_per_loss == 0U || result.prefixes_per_loss > 32U ||
      result.maximum_selected_prefixes > 4'096U ||
      result.continuations_per_prefix == 0U ||
      (result.continuations_per_prefix & 1U) != 0U ||
      result.continuations_per_prefix > 32U || result.shard_count == 0U ||
      result.shard_index >= result.shard_count ||
      !std::isfinite(result.temperature) || result.temperature < 0.0F ||
      result.maximum_generated_turns == 0U ||
      (reanalysis_enabled &&
       (result.reanalysis_work != kRound2TeacherWork ||
        result.verification_work != kRound2VerificationWork ||
        result.reanalysis_samples_per_game == 0U ||
        result.reanalysis_checkpoint.empty() ||
        !valid_sha256(result.reanalysis_artifact_sha256))) ||
      (!reanalysis_enabled &&
       (!result.reanalysis_checkpoint.empty() ||
        !result.reanalysis_artifact_sha256.empty()))) {
    throw std::invalid_argument("invalid live-restart limits or provenance");
  }
  return result;
}

void assert_expected_collector(const RestartArguments &arguments,
                               const CollectorInput &input) {
  if (input.sha256 != arguments.input_sha256 ||
      metadata_value(input, "asserted_source_sha256") !=
          arguments.expected_source_sha256 ||
      metadata_value(input, "arena_manifest_sha256") !=
          arguments.expected_manifest_sha256 ||
      metadata_value(input, "exclusion_registry_sha256") !=
          arguments.expected_exclusion_registry_sha256 ||
      metadata_value(input, "asserted_submission_id") !=
          arguments.expected_submission_id ||
      metadata_value(input, "agent_id") != arguments.expected_agent_id) {
    throw std::invalid_argument(
        "collector TSV does not match the explicit expected identities");
  }
}

void write_restart_record(
    std::ostream &output, const RestartArguments &arguments,
    const CollectorInput &collector, const RestartPrefix &prefix,
    std::size_t continuation, std::size_t game, std::uint64_t game_seed,
    int winner, std::size_t complete_turns,
    const std::vector<Round2Sample> &samples, const LoadedModel &player_one,
    const LoadedModel &player_two, const LoadedModel *teacher,
    std::string_view transcript, const AggregateSearchStats &search_stats) {
  output << "{\"schema\":\"" << kRestartGameSchema << "\",";
  output << "\"feature_schema\":\""
         << papersoccer::jacek_native_model::kFeatureSchema << "\",";
  output << "\"rules\":{\"width\":8,\"height\":10,"
            "\"goal_rule\":\"own-goals-allowed\","
            "\"blocked_rule\":\"mover-loses\"},";
  output << "\"generator\":{\"schema\":\"" << kRestartGeneratorSchema
         << "\",\"action\":\"complete-turn\",\"max_actions\":250,"
            "\"deque_schedule\":\"nine-lifo-one-fifo\",";
  output << "\"search_work\":" << arguments.work
         << ",\"work_unit\":\"maximum-tree-nodes\",";
  output << "\"sampling_temperature\":" << arguments.temperature
         << ",\"temperature_turns\":" << arguments.temperature_turns
         << ",\"temperature_schedule\":\"" << kRestartTemperatureSchedule
         << "\",\"opening_schema\":\"" << kRestartOpeningSchema
         << "\",\"opening_depth\":" << prefix.prefix_turn
         << ",\"opening_transcript\":\"" << prefix.transcript << "\",";
  output << "\"value_target\":\"mover-relative-final-outcome\","
            "\"checkpoint_color_schedule\":\""
         << kRestartColorSchedule << "\",\"producer_sha256\":\""
         << arguments.producer_sha256
         << "\",\"build_provenance_sha256\":\""
         << arguments.build_provenance_sha256 << "\",\"models\":{";
  output << "\"player_one\":";
  write_model_identity(output, player_one);
  output << ",\"player_two\":";
  write_model_identity(output, player_two);
  output << "},\"reanalysis\":{\"selection\":\""
         << kRound2ReanalysisSelection << "\",\"samples_per_game\":"
         << std::count_if(samples.begin(), samples.end(),
                          [](const Round2Sample &sample) {
                            return sample.reanalysis.has_value();
                          })
         << ",\"work\":" << arguments.reanalysis_work
         << ",\"verification_work\":"
         << (arguments.reanalysis_work == 0U ? 0U
                                             : arguments.verification_work)
         << ",\"teacher\":";
  if (teacher == nullptr) output << "null";
  else write_model_identity(output, *teacher);
  output << "},\"source\":{\"input_sha256\":\"" << collector.sha256
         << "\",\"game_id\":\"" << prefix.game_id
         << "\",\"candidate_player\":" << prefix.candidate_player
         << ",\"observed_winner\":" << prefix.observed_winner
         << ",\"observed_turn_count\":" << prefix.observed_turns
         << ",\"prefix_turn\":" << prefix.prefix_turn
         << ",\"prefix_state_id\":\"" << prefix.state_id
         << "\",\"observed_moves_usage\":\"state-construction-only\","
            "\"policy_target\":null,\"input_provenance\":{";
  bool first = true;
  for (const std::string_view key : kRequiredMetadata) {
    if (!first) output << ',';
    first = false;
    output << '\"' << key << "\":\"" << metadata_value(collector, key)
           << '\"';
  }
  output << "}},\"search_stats\":{\"searches\":" << search_stats.searches
         << ",\"expansions\":" << search_stats.expansions
         << ",\"child_evaluations\":" << search_stats.child_evaluations
         << ",\"completed_actions\":" << search_stats.completed_actions
         << ",\"partial_paths\":" << search_stats.partial_paths
         << ",\"tactical_proof_paths\":"
         << search_stats.tactical_proof_paths
         << ",\"generator_truncations\":"
         << search_stats.generator_truncations
         << ",\"tactical_classes_found\":"
         << search_stats.tactical_classes_found
         << ",\"tactical_proof_truncations\":"
         << search_stats.tactical_proof_truncations
         << ",\"tree_cap_searches\":" << search_stats.tree_cap_searches
         << ",\"expansion_cap_searches\":"
         << search_stats.expansion_cap_searches << "}},";
  output << "\"seed\":\"" << game_seed << "\",\"game\":" << game
         << ",\"continuation\":" << continuation
         << ",\"shard_index\":" << arguments.shard_index
         << ",\"shard_count\":" << arguments.shard_count
         << ",\"split_group\":\"native-live-restart:" << collector.sha256
         << ':' << prefix.game_id << "\",\"winner\":" << winner
         << ",\"complete_turns\":" << complete_turns
         << ",\"transcript_schema\":"
            "\"complete-turn-directions-slash/v1\",\"transcript\":\""
         << transcript << "\",\"samples\":[";
  for (std::size_t sample_index = 0; sample_index < samples.size();
       ++sample_index) {
    if (sample_index != 0U) output << ',';
    const Round2Sample &round2 = samples[sample_index];
    const Sample &sample = round2.sample;
    output << "{\"turn\":" << sample.turn << ",\"player\":"
           << (sample.player == ps::Player::One ? 0 : 1)
           << ",\"canonical_state_id\":\"" << sample.canonical_state_id
           << "\",\"active\":[";
    for (std::size_t active = 0; active < sample.active.size(); ++active) {
      if (active != 0U) output << ',';
      output << sample.active[active];
    }
    output << "],\"reflected_state_id\":\"" << sample.reflected_state_id
           << "\",\"reflected_active\":[";
    for (std::size_t active = 0; active < sample.reflected_active.size();
         ++active) {
      if (active != 0U) output << ',';
      output << sample.reflected_active[active];
    }
    output << ']';
    if (round2.reanalysis.has_value()) {
      const ReanalysisDecision &decision = *round2.reanalysis;
      output << ",\"reanalysis\":{\"selection_reason\":\""
             << round2.selection_reason << "\",\"value\":" << decision.value
             << ",\"work\":" << arguments.reanalysis_work
             << ",\"verification_work\":" << arguments.verification_work
             << ",\"operational_interruption\":"
             << (decision.operational_interruption ? "true" : "false")
             << ",\"primary_planned_work_exhaustion\":"
             << (decision.primary_planned_work_exhaustion ? "true" : "false")
             << ",\"verification_planned_work_exhaustion\":"
             << (decision.verification_planned_work_exhaustion ? "true"
                                                               : "false")
             << ",\"primary_generator_sampling_truncations\":"
             << decision.primary_generator_sampling_truncations
             << ",\"verification_generator_sampling_truncations\":"
             << decision.verification_generator_sampling_truncations
             << ",\"primary_proof_sampling_truncations\":"
             << decision.primary_proof_sampling_truncations
             << ",\"verification_proof_sampling_truncations\":"
             << decision.verification_proof_sampling_truncations
             << ",\"action_stable\":"
             << (decision.action_stable ? "true" : "false")
             << ",\"value_delta\":" << decision.value_delta
             << ",\"stable\":" << (decision.stable ? "true" : "false")
             << ",\"exact\":" << (decision.exact ? "true" : "false")
             << '}';
    }
    output << '}';
  }
  output << "]}\n";
}

}  // namespace

#ifndef PAPER_SOCCER_JACEK_NATIVE_RESTART_NO_MAIN
int main(int argc, char **argv) {
  try {
    const RestartArguments arguments = parse_restart_arguments(argc, argv);
    const CollectorInput collector = read_collector(arguments.input);
    assert_expected_collector(arguments, collector);
    const std::vector<RestartPrefix> prefixes = limit_restart_prefixes(
        validate_and_select_prefixes(collector, arguments.prefixes_per_loss),
        arguments.maximum_selected_prefixes);
    if (prefixes.size() > 65'536U / arguments.continuations_per_prefix) {
      throw std::invalid_argument("live-restart record plan exceeds 65536 games");
    }
    if (std::filesystem::exists(arguments.output)) {
      throw std::runtime_error("refusing to overwrite restart corpus output");
    }
    const LoadedModel base_player_one = load_checkpoint(
        arguments.player_one_checkpoint, arguments.player_one_artifact_sha256);
    const LoadedModel base_player_two = load_checkpoint(
        arguments.player_two_checkpoint, arguments.player_two_artifact_sha256);
    const LoadedModel teacher = load_checkpoint(
        arguments.reanalysis_checkpoint, arguments.reanalysis_artifact_sha256);
    const LoadedModel *teacher_pointer =
        arguments.reanalysis_work == 0U ? nullptr : &teacher;
    std::ofstream output(arguments.output);
    if (!output) throw std::runtime_error("could not open restart corpus output");

    std::size_t completed = 0;
    for (std::size_t prefix_index = 0; prefix_index < prefixes.size();
         ++prefix_index) {
      const RestartPrefix &prefix = prefixes[prefix_index];
      for (std::size_t continuation = 0;
           continuation < arguments.continuations_per_prefix;
           ++continuation) {
        const std::size_t game =
            prefix_index * arguments.continuations_per_prefix + continuation;
        if (game % arguments.shard_count != arguments.shard_index) continue;
        const std::uint64_t game_seed =
            arguments.seed + game * 0x9e3779b97f4a7c15ULL;
        const bool swap_models = (continuation & 1U) != 0U;
        const LoadedModel &player_one =
            swap_models ? base_player_two : base_player_one;
        const LoadedModel &player_two =
            swap_models ? base_player_one : base_player_two;
        SplitMix64 random(game_seed ^ 0xd1b54a32d192ed03ULL);
        ps::GameState state = prefix.state;
        std::string transcript = prefix.transcript;
        std::vector<Round2Sample> samples;
        AggregateSearchStats search_stats;
        std::size_t generated = 0;
        for (; generated < arguments.maximum_generated_turns &&
               !ps::is_terminal(state);
             ++generated) {
          const std::size_t turn = prefix.prefix_turn + generated;
          std::vector<std::uint16_t> active = active_features(state);
          std::vector<std::uint16_t> reflected =
              active_features(reflected_state(state));
          Sample sample{turn, state.to_move, state, active, feature_id(active),
                        reflected, feature_id(reflected)};
          const native::QuantizedModel *turn_model =
              state.to_move == ps::Player::One ? player_one.model.get()
                                                : player_two.model.get();
          const native::SearchResult search = native::choose_complete_turn(
              state, search_config(game_seed ^ random.next(), turn_model,
                                   arguments.work));
          search_stats.observe(search.stats);
          const float gameplay_value = std::clamp(search.value, -1.0F, 1.0F);
          samples.push_back(Round2Sample{
              std::move(sample), gameplay_value, 0.0F,
              std::abs(gameplay_value), {}, std::nullopt});
          const std::string action =
              generated < arguments.temperature_turns
                  ? sampled_action(search, arguments.temperature, random)
                  : search.encoded;
          if (!transcript.empty()) transcript.push_back('/');
          transcript += action;
          native::apply_encoded_turn(state, action);
        }
        if (!ps::is_terminal(state)) {
          throw std::runtime_error(
              "restart self-play exceeded generated complete-turn cap");
        }
        const ps::Player winner_player = *ps::winner(state);
        const int winner = winner_player == ps::Player::One ? 0 : 1;
        const std::size_t complete_turns = prefix.prefix_turn + generated;
        samples = select_evenly_round2(std::move(samples),
                                       arguments.samples_per_game);
        for (Round2Sample &sample : samples) {
          const float outcome =
              sample.sample.player == winner_player ? 1.0F : -1.0F;
          sample.outcome_error = std::abs(sample.gameplay_value - outcome);
        }
        if (teacher_pointer != nullptr) {
          for (const auto &[sample_index, reason] : select_reanalysis_indices(
                   samples, arguments.reanalysis_samples_per_game)) {
            Round2Sample &sample = samples[sample_index];
            sample.selection_reason = reason;
            const std::uint64_t teacher_seed =
                game_seed ^
                (sample.sample.turn * 0x9e3779b97f4a7c15ULL) ^
                0xa4093822299f31d0ULL;
            const native::SearchResult primary = native::choose_complete_turn(
                sample.sample.state,
                search_config(teacher_seed, teacher_pointer->model.get(),
                              arguments.reanalysis_work));
            const native::SearchResult verification =
                native::choose_complete_turn(
                    sample.sample.state,
                    search_config(teacher_seed, teacher_pointer->model.get(),
                                  arguments.verification_work));
            sample.reanalysis = classify_reanalysis(
                primary, verification, sample.sample.player);
          }
        }
        write_restart_record(
            output, arguments, collector, prefix, continuation, game,
            game_seed, winner, complete_turns, samples, player_one, player_two,
            teacher_pointer, transcript, search_stats);
        ++completed;
      }
    }
    std::cerr << "completed " << completed << " live-restart shard games\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Jacek-native live restart: " << error.what() << '\n';
    return 1;
  }
}
#endif
