#include <array>
#include <optional>
#include <tuple>

// Round one is a frozen production dependency.  Reuse its rules, feature,
// checkpoint, and self-play helpers without changing the source that produced
// the promoted model.
#define main jacek_native_round1_main
#include "jacek_native_selfplay.cpp"
#undef main

namespace {

constexpr std::string_view kRound2GameSchema =
    "papersoccer.jacek-native-game/v2";
constexpr std::string_view kRound2GeneratorSchema =
    "jacek-native-complete-turn-bfm/v2";
constexpr std::string_view kRound2ColorSchedule =
    "paired-opening-depth-then-swap-checkpoints/v2";
constexpr std::string_view kRound2ReanalysisSelection =
    "alternating-hard-error-and-low-confidence/v1";
constexpr std::uint64_t kRound2TeacherWork = 30'000;
constexpr std::uint64_t kRound2VerificationWork = 100'000;

struct Round2Arguments {
  std::string output;
  std::string checkpoint;
  std::string player_one_checkpoint;
  std::string player_two_checkpoint;
  std::string player_one_artifact_sha256;
  std::string player_two_artifact_sha256;
  std::string reanalysis_checkpoint;
  std::string reanalysis_artifact_sha256;
  std::string producer_sha256;
  std::string build_provenance_sha256;
  std::size_t games{kDefaultGames};
  std::uint64_t seed{kDefaultSeed};
  std::uint64_t work{kDefaultWork};
  std::uint64_t reanalysis_work{};
  std::uint64_t verification_work{kRound2VerificationWork};
  std::size_t samples_per_game{kDefaultSamplesPerGame};
  std::size_t reanalysis_samples_per_game{12};
  std::size_t shard_index{};
  std::size_t shard_count{1};
  float temperature{3.0F};
  std::size_t temperature_turns{12};
  std::vector<std::size_t> opening_depths{0, 4, 8, 12};
  std::size_t maximum_complete_turns{kMaximumCompleteTurns};
};

struct ReanalysisDecision {
  float value{};
  bool operational_interruption{};
  bool primary_planned_work_exhaustion{};
  bool verification_planned_work_exhaustion{};
  std::uint64_t primary_generator_sampling_truncations{};
  std::uint64_t verification_generator_sampling_truncations{};
  std::uint64_t primary_proof_sampling_truncations{};
  std::uint64_t verification_proof_sampling_truncations{};
  bool action_stable{};
  float value_delta{};
  bool stable{};
  bool exact{};
};

struct Round2Sample {
  Sample sample;
  float gameplay_value{};
  float outcome_error{};
  float uncertainty{};
  std::string selection_reason;
  std::optional<ReanalysisDecision> reanalysis;
};

struct Round2Schedule {
  std::size_t opening_pair_index{};
  std::size_t opening_depth{};
  std::uint64_t opening_pair_seed{};
  bool swap_models{};
};

Round2Schedule schedule_round2_game(const Round2Arguments &arguments,
                                    std::size_t game) {
  const std::size_t pair = game / 2U;
  return Round2Schedule{
      pair,
      arguments.opening_depths[pair % arguments.opening_depths.size()],
      arguments.seed + pair * 0xd1b54a32d192ed03ULL,
      (game & 1U) != 0U};
}

std::uint64_t parse_round2_unsigned(std::string_view text,
                                    std::string_view label) {
  if (text.empty() || !std::all_of(text.begin(), text.end(), [](char value) {
        return value >= '0' && value <= '9';
      })) {
    throw std::invalid_argument(std::string(label) +
                                " must be an unsigned decimal integer");
  }
  return parse_unsigned(text, label);
}

std::vector<std::size_t> parse_round2_depths(std::string_view text) {
  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t separator = text.find(',', start);
    const std::size_t stop = separator == std::string_view::npos
                                 ? text.size()
                                 : separator;
    if (stop == start) {
      throw std::invalid_argument("opening depths contain an empty item");
    }
    result.push_back(parse_round2_unsigned(
        text.substr(start, stop - start), "opening depth"));
    if (separator == std::string_view::npos) break;
    start = separator + 1;
  }
  return result;
}

[[maybe_unused]] Round2Arguments parse_round2_arguments(int argc, char **argv) {
  Round2Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string_view value = argv[++index];
    if (option == "--output") result.output = value;
    else if (option == "--checkpoint") result.checkpoint = value;
    else if (option == "--player-one-checkpoint") {
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
    } else if (option == "--games") {
      result.games = parse_round2_unsigned(value, "games");
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
    } else if (option == "--shard-index") {
      result.shard_index = parse_round2_unsigned(value, "shard index");
    } else if (option == "--shard-count") {
      result.shard_count = parse_round2_unsigned(value, "shard count");
    } else if (option == "--temperature") {
      result.temperature = parse_float(value, "temperature");
    } else if (option == "--temperature-turns") {
      result.temperature_turns =
          parse_round2_unsigned(value, "temperature turns");
    } else if (option == "--opening-depths") {
      result.opening_depths = parse_round2_depths(value);
    } else if (option == "--max-complete-turns") {
      result.maximum_complete_turns =
          parse_round2_unsigned(value, "maximum complete turns");
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  if (result.output.empty()) {
    throw std::invalid_argument("--output is required");
  }
  if (!result.checkpoint.empty()) {
    if (result.player_one_checkpoint.empty()) {
      result.player_one_checkpoint = result.checkpoint;
    }
    if (result.player_two_checkpoint.empty()) {
      result.player_two_checkpoint = result.checkpoint;
    }
  }
  const bool has_player_one = !result.player_one_checkpoint.empty();
  const bool has_player_two = !result.player_two_checkpoint.empty();
  const bool has_reanalysis = result.reanalysis_work != 0;
  if (result.games == 0 || result.work < 2 || result.samples_per_game == 0 ||
      result.samples_per_game > 100 || result.shard_count == 0 ||
      result.shard_index >= result.shard_count || result.temperature < 0.0F ||
      result.maximum_complete_turns == 0 || !has_player_one || !has_player_two ||
      result.reanalysis_samples_per_game > result.samples_per_game ||
      (has_reanalysis &&
       (result.reanalysis_work != kRound2TeacherWork ||
        result.verification_work != kRound2VerificationWork ||
        result.reanalysis_samples_per_game == 0 ||
        result.reanalysis_checkpoint.empty())) ||
      (!has_reanalysis && (!result.reanalysis_checkpoint.empty() ||
                           !result.reanalysis_artifact_sha256.empty()))) {
    throw std::invalid_argument("invalid round-two self-play limits");
  }
  const std::size_t schedule_cycle = 2U * result.opening_depths.size();
  if (result.games % schedule_cycle != 0) {
    throw std::invalid_argument(
        "games must contain complete depth/color schedule cycles");
  }
  if (std::any_of(result.opening_depths.begin(), result.opening_depths.end(),
                  [&](std::size_t depth) {
                    return depth >= result.maximum_complete_turns;
                  })) {
    throw std::invalid_argument(
        "opening depths must be below the maximum complete-turn count");
  }
  if (!valid_sha256(result.producer_sha256) ||
      !valid_sha256(result.build_provenance_sha256) ||
      (has_player_one &&
       (!valid_sha256(result.player_one_artifact_sha256) ||
        !valid_sha256(result.player_two_artifact_sha256))) ||
      (has_reanalysis &&
       !valid_sha256(result.reanalysis_artifact_sha256))) {
    throw std::invalid_argument(
        "provenance SHA-256 values must be lowercase hex");
  }
  return result;
}

[[maybe_unused]] std::vector<Round2Sample> select_evenly_round2(
    std::vector<Round2Sample> samples, std::size_t maximum) {
  if (samples.size() <= maximum) return samples;
  if (maximum == 1) return {std::move(samples.front())};
  std::vector<Round2Sample> result;
  result.reserve(maximum);
  for (std::size_t index = 0; index < maximum; ++index) {
    const std::size_t source = index * (samples.size() - 1) / (maximum - 1);
    result.push_back(std::move(samples[source]));
  }
  return result;
}

std::vector<std::pair<std::size_t, std::string>>
select_reanalysis_indices(const std::vector<Round2Sample> &samples,
                          std::size_t maximum) {
  maximum = std::min(maximum, samples.size());
  std::vector<std::size_t> hard(samples.size());
  std::vector<std::size_t> uncertain(samples.size());
  for (std::size_t index = 0; index < samples.size(); ++index) {
    hard[index] = index;
    uncertain[index] = index;
  }
  const auto stable_tail = [&](std::size_t left, std::size_t right) {
    return std::tie(samples[left].sample.turn, left) <
           std::tie(samples[right].sample.turn, right);
  };
  std::stable_sort(hard.begin(), hard.end(), [&](std::size_t left,
                                                 std::size_t right) {
    if (samples[left].outcome_error != samples[right].outcome_error) {
      return samples[left].outcome_error > samples[right].outcome_error;
    }
    if (samples[left].uncertainty != samples[right].uncertainty) {
      return samples[left].uncertainty < samples[right].uncertainty;
    }
    return stable_tail(left, right);
  });
  std::stable_sort(uncertain.begin(), uncertain.end(),
                   [&](std::size_t left, std::size_t right) {
    if (samples[left].uncertainty != samples[right].uncertainty) {
      return samples[left].uncertainty < samples[right].uncertainty;
    }
    if (samples[left].outcome_error != samples[right].outcome_error) {
      return samples[left].outcome_error > samples[right].outcome_error;
    }
    return stable_tail(left, right);
  });
  std::vector<std::pair<std::size_t, std::string>> result;
  std::vector<bool> selected(samples.size(), false);
  result.reserve(maximum);
  std::array<std::size_t, 2> cursor{};
  for (std::size_t turn = 0; result.size() < maximum; ++turn) {
    const bool choose_hard = (turn & 1U) == 0U;
    const std::vector<std::size_t> &ordered = choose_hard ? hard : uncertain;
    std::size_t &position = cursor[choose_hard ? 0U : 1U];
    while (position < ordered.size() && selected[ordered[position]]) ++position;
    if (position == ordered.size()) {
      const std::vector<std::size_t> &other = choose_hard ? uncertain : hard;
      std::size_t &other_position = cursor[choose_hard ? 1U : 0U];
      while (other_position < other.size() && selected[other[other_position]]) {
        ++other_position;
      }
      if (other_position == other.size()) break;
      const std::size_t index = other[other_position++];
      selected[index] = true;
      result.emplace_back(index, choose_hard ? "uncertain" : "hard");
      continue;
    }
    const std::size_t index = ordered[position++];
    selected[index] = true;
    result.emplace_back(index, choose_hard ? "hard" : "uncertain");
  }
  return result;
}

// Fixed-work reanalysis is complete when the configured capped BFM finishes
// without a deadline interruption.  Its 250-action, partial-path, and bounded
// proof sampling caps are architectural choices, not operational failures; we
// disclose their counts separately in every reanalysis label.
bool operationally_interrupted(const native::SearchResult &result) {
  return result.stats.deadline_reached;
}

bool planned_work_exhausted(const native::SearchResult &result) {
  return result.stats.tree_cap_reached || result.stats.expansion_cap_reached;
}

ReanalysisDecision classify_reanalysis(
    const native::SearchResult &primary,
    const native::SearchResult &verification,
    ps::Player sample_player) {
  const float first_value = std::clamp(primary.value, -1.0F, 1.0F);
  const float second_value = std::clamp(verification.value, -1.0F, 1.0F);
  const bool exact = verification.solved &&
                     verification.solved_winner.has_value();
  const bool operational = operationally_interrupted(primary) ||
                           operationally_interrupted(verification);
  const bool action_stable = primary.encoded == verification.encoded;
  const float value_delta = std::abs(first_value - second_value);
  const float selected_value = exact
      ? (*verification.solved_winner == sample_player ? 1.0F : -1.0F)
      : second_value;
  return ReanalysisDecision{
      selected_value,
      operational,
      planned_work_exhausted(primary),
      planned_work_exhausted(verification),
      primary.stats.generator_truncations,
      verification.stats.generator_truncations,
      primary.stats.proof_truncations,
      verification.stats.proof_truncations,
      action_stable,
      value_delta,
      exact || (!operational && action_stable && value_delta <= 0.05F),
      exact};
}

void write_model_identity(std::ostream &output, const LoadedModel &loaded) {
  output << "{\"model_sha256\":\""
         << (loaded.model
                 ? loaded.model_sha256
                 : std::string(
                       papersoccer::jacek_native_model::kModelSha256))
         << "\",\"packed_sha256\":\""
         << (loaded.model
                 ? loaded.packed_sha256
                 : std::string(
                       papersoccer::jacek_native_model::kPackedSha256))
         << "\",\"artifact_sha256\":\""
         << (loaded.artifact_sha256.empty()
                 ? std::string(
                       papersoccer::jacek_native_model::kModelSha256)
                 : loaded.artifact_sha256)
         << "\"}";
}

[[maybe_unused]] void write_round2_record(
    std::ostream &output, const Round2Arguments &arguments,
    std::size_t game, std::uint64_t game_seed, int winner,
    std::size_t complete_turns, const std::vector<Round2Sample> &samples,
    const LoadedModel &player_one, const LoadedModel &player_two,
    const LoadedModel *teacher, const Opening &opening,
    std::size_t opening_pair_index, std::string_view transcript,
    const AggregateSearchStats &search_stats) {
  output << "{\"schema\":\"" << kRound2GameSchema << "\",";
  output << "\"feature_schema\":\""
         << papersoccer::jacek_native_model::kFeatureSchema << "\",";
  output << "\"rules\":{\"width\":8,\"height\":10,"
            "\"goal_rule\":\"own-goals-allowed\","
            "\"blocked_rule\":\"mover-loses\"},";
  output << "\"generator\":{\"schema\":\"" << kRound2GeneratorSchema
         << "\",\"action\":\"complete-turn\",\"max_actions\":250,"
            "\"deque_schedule\":\"nine-lifo-one-fifo\",";
  output << "\"search_work\":" << arguments.work << ','
         << "\"work_unit\":\"maximum-tree-nodes\",";
  output << "\"sampling_temperature\":" << arguments.temperature << ','
         << "\"temperature_turns\":" << arguments.temperature_turns << ','
         << "\"temperature_schedule\":\"" << kTemperatureSchedule << "\",";
  output << "\"opening_schema\":\"" << kOpeningSchema << "\","
         << "\"opening_depth\":" << opening.depth << ','
         << "\"opening_pair_index\":" << opening_pair_index << ','
         << "\"opening_seed\":\"" << opening.seed << "\","
         << "\"opening_retry\":" << opening.retry << ','
         << "\"opening_transcript\":\"" << opening.transcript << "\",";
  output << "\"value_target\":\"mover-relative-final-outcome\","
         << "\"checkpoint_color_schedule\":\"" << kRound2ColorSchedule
         << "\",\"producer_sha256\":\"" << arguments.producer_sha256
         << "\",\"build_provenance_sha256\":\""
         << arguments.build_provenance_sha256 << "\",";
  output << "\"models\":{\"player_one\":";
  write_model_identity(output, player_one);
  output << ",\"player_two\":";
  write_model_identity(output, player_two);
  output << "},\"reanalysis\":{"
         << "\"selection\":\"" << kRound2ReanalysisSelection << "\","
         << "\"samples_per_game\":"
         << std::count_if(samples.begin(), samples.end(),
                          [](const Round2Sample &sample) {
                            return sample.reanalysis.has_value();
                          })
         << ",\"work\":" << arguments.reanalysis_work
         << ",\"verification_work\":"
         << (arguments.reanalysis_work == 0 ? 0 : arguments.verification_work)
         << ",\"teacher\":";
  if (teacher == nullptr) output << "null";
  else write_model_identity(output, *teacher);
  output << "},\"search_stats\":{"
         << "\"searches\":" << search_stats.searches << ','
         << "\"expansions\":" << search_stats.expansions << ','
         << "\"child_evaluations\":" << search_stats.child_evaluations << ','
         << "\"completed_actions\":" << search_stats.completed_actions << ','
         << "\"partial_paths\":" << search_stats.partial_paths << ','
         << "\"tactical_proof_paths\":" << search_stats.tactical_proof_paths
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
         << ",\"shard_index\":" << arguments.shard_index
         << ",\"shard_count\":" << arguments.shard_count
         << ",\"split_group\":\"native-round2:" << game_seed << ':' << game
         << "\",\"winner\":" << winner
         << ",\"complete_turns\":" << complete_turns
         << ",\"transcript_schema\":"
            "\"complete-turn-directions-slash/v1\","
         << "\"transcript\":\"" << transcript << "\",\"samples\":[";
  for (std::size_t sample_index = 0; sample_index < samples.size();
       ++sample_index) {
    const Round2Sample &round2 = samples[sample_index];
    const Sample &sample = round2.sample;
    if (sample_index != 0) output << ',';
    output << "{\"turn\":" << sample.turn << ",\"player\":"
           << (sample.player == ps::Player::One ? 0 : 1)
           << ",\"canonical_state_id\":\"" << sample.canonical_state_id
           << "\",\"active\":[";
    for (std::size_t active = 0; active < sample.active.size(); ++active) {
      if (active != 0) output << ',';
      output << sample.active[active];
    }
    output << "],\"reflected_state_id\":\"" << sample.reflected_state_id
           << "\",\"reflected_active\":[";
    for (std::size_t active = 0; active < sample.reflected_active.size();
         ++active) {
      if (active != 0) output << ',';
      output << sample.reflected_active[active];
    }
    output << ']';
    if (round2.reanalysis.has_value()) {
      const ReanalysisDecision &decision = *round2.reanalysis;
      output << ",\"reanalysis\":{\"selection_reason\":\""
             << round2.selection_reason << "\",\"value\":"
             << decision.value << ",\"work\":"
             << arguments.reanalysis_work << ",\"verification_work\":"
             << arguments.verification_work
             << ",\"operational_interruption\":"
             << (decision.operational_interruption ? "true" : "false")
             << ",\"primary_planned_work_exhaustion\":"
             << (decision.primary_planned_work_exhaustion ? "true" : "false")
             << ",\"verification_planned_work_exhaustion\":"
             << (decision.verification_planned_work_exhaustion
                     ? "true" : "false")
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

#ifndef PAPER_SOCCER_JACEK_NATIVE_ROUND2_NO_MAIN
int main(int argc, char **argv) {
  try {
    const Round2Arguments arguments = parse_round2_arguments(argc, argv);
    if (std::filesystem::exists(arguments.output)) {
      throw std::runtime_error("refusing to overwrite existing corpus output");
    }
    std::ofstream output(arguments.output);
    if (!output) throw std::runtime_error("could not open corpus output");
    // Preserve the exact float comparisons behind stable/value-delta fields
    // when the JSON is revalidated in Python.
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    const LoadedModel base_player_one = load_checkpoint(
        arguments.player_one_checkpoint, arguments.player_one_artifact_sha256);
    const LoadedModel base_player_two = load_checkpoint(
        arguments.player_two_checkpoint, arguments.player_two_artifact_sha256);
    const LoadedModel teacher = load_checkpoint(
        arguments.reanalysis_checkpoint, arguments.reanalysis_artifact_sha256);
    const LoadedModel *teacher_pointer =
        arguments.reanalysis_work == 0 ? nullptr : &teacher;
    std::size_t completed = 0;
    for (std::size_t game = arguments.shard_index; game < arguments.games;
         game += arguments.shard_count) {
      const std::uint64_t game_seed =
          arguments.seed + game * 0x9e3779b97f4a7c15ULL;
      const Round2Schedule schedule = schedule_round2_game(arguments, game);
      const LoadedModel &player_one =
          schedule.swap_models ? base_player_two : base_player_one;
      const LoadedModel &player_two =
          schedule.swap_models ? base_player_one : base_player_two;
      SplitMix64 random(game_seed ^ 0xd1b54a32d192ed03ULL);
      ps::RulesConfig rules;
      rules.width = 8;
      rules.height = 10;
      rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
      rules.blocked_rule = ps::BlockedRule::MoverLoses;
      const Opening opening =
          make_opening(rules, schedule.opening_pair_seed,
                       schedule.opening_depth);
      ps::GameState state = opening.state;
      std::string transcript = opening.transcript;
      std::vector<Round2Sample> samples;
      AggregateSearchStats search_stats;
      std::size_t turn = schedule.opening_depth;
      for (; turn < arguments.maximum_complete_turns && !ps::is_terminal(state);
           ++turn) {
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
            turn < arguments.temperature_turns
                ? sampled_action(search, arguments.temperature, random)
                : search.encoded;
        if (!transcript.empty()) transcript.push_back('/');
        transcript += action;
        native::apply_encoded_turn(state, action);
      }
      if (!ps::is_terminal(state)) {
        throw std::runtime_error("self-play game exceeded complete-turn cap");
      }
      const ps::Player winner_player = *ps::winner(state);
      const int winner = winner_player == ps::Player::One ? 0 : 1;
      const std::size_t complete_turns = turn;
      samples = select_evenly_round2(
          std::move(samples), arguments.samples_per_game);
      for (Round2Sample &sample : samples) {
        const float outcome =
            sample.sample.player == winner_player ? 1.0F : -1.0F;
        sample.outcome_error = std::abs(sample.gameplay_value - outcome);
      }
      if (teacher_pointer != nullptr) {
        for (const auto &[index, reason] : select_reanalysis_indices(
                 samples, arguments.reanalysis_samples_per_game)) {
          Round2Sample &sample = samples[index];
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
      write_round2_record(
          output, arguments, game, game_seed, winner, complete_turns, samples,
          player_one, player_two, teacher_pointer, opening,
          schedule.opening_pair_index, transcript, search_stats);
      ++completed;
      if (completed % 10 == 0) {
        std::cerr << "completed " << completed << " shard games\n";
      }
    }
    std::cerr << "completed " << completed << " shard games\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Jacek-native round-two self-play: " << error.what() << '\n';
    return 1;
  }
}
#endif
