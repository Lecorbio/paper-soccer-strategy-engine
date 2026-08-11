#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#include "../submissions/codingame/bots/jacek_native_bfm/bot.cpp"

namespace ps = papersoccer;
namespace native = papersoccer::jacek_native_bfm;

namespace {

constexpr std::string_view kRuntimeSchema =
    "papersoccer.jacek-native-runtime-model/v1";
constexpr std::size_t kDefaultGames = 10'000;
constexpr std::uint64_t kDefaultSeed = 73194721ULL;
constexpr std::uint64_t kDefaultWork = 2'048;
constexpr std::size_t kDefaultSamplesPerGame = 100;
constexpr std::size_t kMaximumCompleteTurns = 384;
constexpr std::string_view kOpeningSchema =
    "deterministic-procedural-complete-turn-prefix/v1";
constexpr std::string_view kTemperatureSchedule =
    "absolute-complete-turn-index-before-cutoff/v1";

struct Arguments {
  std::string output;
  std::string checkpoint;
  std::string player_one_checkpoint;
  std::string player_two_checkpoint;
  std::string player_one_artifact_sha256;
  std::string player_two_artifact_sha256;
  std::string producer_sha256;
  std::string build_provenance_sha256;
  std::size_t games{kDefaultGames};
  std::uint64_t seed{kDefaultSeed};
  std::uint64_t work{kDefaultWork};
  std::uint64_t reanalysis_work{};
  std::size_t samples_per_game{kDefaultSamplesPerGame};
  std::size_t shard_index{};
  std::size_t shard_count{1};
  float temperature{3.0F};
  std::size_t temperature_turns{12};
  std::vector<std::size_t> opening_depths{0};
  std::size_t maximum_complete_turns{kMaximumCompleteTurns};
};

struct LoadedModel {
  std::unique_ptr<native::QuantizedModel> model;
  std::string model_sha256;
  std::string packed_sha256;
  std::string artifact_sha256;
};

struct Sample {
  std::size_t turn{};
  ps::Player player{ps::Player::One};
  ps::GameState state;
  std::vector<std::uint16_t> active;
  std::string canonical_state_id;
  std::vector<std::uint16_t> reflected_active;
  std::string reflected_state_id;
};

struct Opening {
  ps::GameState state;
  std::size_t depth{};
  std::uint64_t seed{};
  std::size_t retry{};
  std::string transcript;
};

struct AggregateSearchStats {
  std::uint64_t searches{};
  std::uint64_t expansions{};
  std::uint64_t child_evaluations{};
  std::uint64_t completed_actions{};
  std::uint64_t partial_paths{};
  std::uint64_t tactical_proof_paths{};
  std::uint64_t generator_truncations{};
  std::uint64_t tactical_classes_found{};
  std::uint64_t tactical_proof_truncations{};
  std::uint64_t tree_cap_searches{};
  std::uint64_t expansion_cap_searches{};

  void observe(const native::SearchStats &stats) noexcept {
    ++searches;
    expansions += stats.expansions;
    child_evaluations += stats.child_evaluations;
    completed_actions += stats.completed_actions;
    partial_paths += stats.generator_partial_paths;
    tactical_proof_paths += stats.proof_paths;
    generator_truncations += stats.generator_truncations;
    tactical_classes_found += stats.proof_classes;
    tactical_proof_truncations += stats.proof_truncations;
    tree_cap_searches += stats.tree_cap_reached ? 1U : 0U;
    expansion_cap_searches += stats.expansion_cap_reached ? 1U : 0U;
  }
};

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}

  std::uint64_t next() noexcept {
    std::uint64_t result = (state_ += 0x9e3779b97f4a7c15ULL);
    result = (result ^ (result >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    result = (result ^ (result >> 27U)) * 0x94d049bb133111ebULL;
    return result ^ (result >> 31U);
  }

  double unit() noexcept {
    return static_cast<double>(next() >> 11U) *
           (1.0 / 9007199254740992.0);
  }

  std::size_t index(std::size_t bound) noexcept {
    if (bound < 2) return 0;
    const std::uint64_t unsigned_bound = static_cast<std::uint64_t>(bound);
    const std::uint64_t threshold =
        static_cast<std::uint64_t>(-unsigned_bound) % unsigned_bound;
    for (;;) {
      const std::uint64_t value = next();
      if (value >= threshold) {
        return static_cast<std::size_t>(value % unsigned_bound);
      }
    }
  }

 private:
  std::uint64_t state_{};
};

std::uint64_t parse_unsigned(std::string_view text, std::string_view label) {
  try {
    std::size_t consumed = 0;
    const auto value = std::stoull(std::string(text), &consumed);
    if (consumed != text.size()) throw std::invalid_argument("trailing");
    return value;
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(label) +
                                " must be an unsigned integer");
  }
}

float parse_float(std::string_view text, std::string_view label) {
  try {
    std::size_t consumed = 0;
    const float value = std::stof(std::string(text), &consumed);
    if (consumed != text.size() || !std::isfinite(value)) {
      throw std::invalid_argument("invalid");
    }
    return value;
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string(label) +
                                " must be a finite number");
  }
}

bool valid_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

std::vector<std::size_t> parse_depths(std::string_view text) {
  std::vector<std::size_t> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t separator = text.find(',', start);
    const std::size_t stop = separator == std::string_view::npos
                                 ? text.size() : separator;
    if (stop == start) throw std::invalid_argument("opening depths contain an empty item");
    result.push_back(parse_unsigned(text.substr(start, stop - start),
                                    "opening depth"));
    if (separator == std::string_view::npos) break;
    start = separator + 1;
  }
  if (result.empty()) throw std::invalid_argument("opening depths are empty");
  return result;
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("every option requires a value");
    }
    const std::string_view value = argv[++index];
    if (option == "--output") result.output = value;
    else if (option == "--checkpoint") result.checkpoint = value;
    else if (option == "--player-one-checkpoint") result.player_one_checkpoint = value;
    else if (option == "--player-two-checkpoint") result.player_two_checkpoint = value;
    else if (option == "--player-one-artifact-sha256") {
      result.player_one_artifact_sha256 = value;
    } else if (option == "--player-two-artifact-sha256") {
      result.player_two_artifact_sha256 = value;
    } else if (option == "--producer-sha256") result.producer_sha256 = value;
    else if (option == "--build-provenance-sha256") {
      result.build_provenance_sha256 = value;
    } else if (option == "--games") {
      result.games = parse_unsigned(value, "games");
    }
    else if (option == "--seed") result.seed = parse_unsigned(value, "seed");
    else if (option == "--work") result.work = parse_unsigned(value, "work");
    else if (option == "--reanalysis-work") {
      result.reanalysis_work = parse_unsigned(value, "reanalysis work");
    } else if (option == "--samples-per-game") {
      result.samples_per_game = parse_unsigned(value, "samples per game");
    } else if (option == "--shard-index") {
      result.shard_index = parse_unsigned(value, "shard index");
    } else if (option == "--shard-count") {
      result.shard_count = parse_unsigned(value, "shard count");
    } else if (option == "--temperature") {
      result.temperature = parse_float(value, "temperature");
    } else if (option == "--temperature-turns") {
      result.temperature_turns = parse_unsigned(value, "temperature turns");
    } else if (option == "--opening-depths") {
      result.opening_depths = parse_depths(value);
    } else if (option == "--max-complete-turns") {
      result.maximum_complete_turns =
          parse_unsigned(value, "maximum complete turns");
    } else {
      throw std::invalid_argument("unknown option " + std::string(option));
    }
  }
  if (result.output.empty()) throw std::invalid_argument("--output is required");
  if (!result.checkpoint.empty()) {
    if (result.player_one_checkpoint.empty()) {
      result.player_one_checkpoint = result.checkpoint;
    }
    if (result.player_two_checkpoint.empty()) {
      result.player_two_checkpoint = result.checkpoint;
    }
  }
  if (result.games == 0 || result.work < 2 || result.samples_per_game == 0 ||
      result.samples_per_game > 100 || result.shard_count == 0 ||
      result.shard_index >= result.shard_count || result.temperature < 0.0F ||
      result.maximum_complete_turns == 0 ||
      (result.reanalysis_work != 0 &&
       (result.reanalysis_work < 30'000 || result.reanalysis_work > 100'000))) {
    throw std::invalid_argument("invalid self-play limits");
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
      (!result.player_one_artifact_sha256.empty() &&
       !valid_sha256(result.player_one_artifact_sha256)) ||
      (!result.player_two_artifact_sha256.empty() &&
       !valid_sha256(result.player_two_artifact_sha256))) {
    throw std::invalid_argument("provenance SHA-256 values must be lowercase hex");
  }
  return result;
}

LoadedModel load_checkpoint(const std::string &path,
                            const std::string &artifact_sha256) {
  if (path.empty()) return LoadedModel{nullptr, {}, {}, artifact_sha256};
  std::ifstream input(path);
  if (!input) throw std::runtime_error("could not open runtime checkpoint");
  std::string schema;
  std::string model_schema;
  std::string feature_schema;
  std::string model_sha;
  std::string packed_sha;
  std::string scales;
  std::string packed;
  std::getline(input, schema);
  std::getline(input, model_schema);
  std::getline(input, feature_schema);
  std::getline(input, model_sha);
  std::getline(input, packed_sha);
  std::getline(input, scales);
  std::getline(input, packed);
  std::string trailing;
  while (std::getline(input, trailing)) {
    if (!trailing.empty()) throw std::invalid_argument("runtime checkpoint has trailing data");
  }
  if (schema != kRuntimeSchema) {
    throw std::invalid_argument("unexpected runtime checkpoint schema");
  }
  if (!valid_sha256(model_sha) || !valid_sha256(packed_sha) ||
      !valid_sha256(artifact_sha256)) {
    throw std::invalid_argument("runtime checkpoint provenance is incomplete");
  }
  std::istringstream scale_input(scales);
  float scale_one = 0.0F;
  float scale_two = 0.0F;
  float scale_three = 0.0F;
  if (!(scale_input >> scale_one >> scale_two >> scale_three) ||
      scale_input.peek() != std::char_traits<char>::eof()) {
    throw std::invalid_argument("invalid runtime checkpoint scales");
  }
  auto model = std::make_unique<native::QuantizedModel>(
      packed, scale_one, scale_two, scale_three, model_schema,
      feature_schema, model_sha, model_sha, false, packed_sha);
  if (model->packed_sha256() != packed_sha) {
    throw std::invalid_argument("runtime checkpoint packed SHA-256 mismatch");
  }
  return LoadedModel{std::move(model), model_sha, packed_sha, artifact_sha256};
}

std::vector<std::uint16_t> active_features(const ps::GameState &state) {
  const native::DenseFeatures dense = native::encode_features(state);
  std::vector<std::uint16_t> result;
  result.reserve(native::kVertices + native::kEdgeInputs);
  for (std::size_t index = 0; index < dense.size(); ++index) {
    if (dense[index] != 0) result.push_back(static_cast<std::uint16_t>(index));
  }
  return result;
}

ps::Point reflect_point(const ps::RulesConfig &config, ps::Point point) {
  return {config.width - point.x, point.y};
}

ps::GameState reflected_state(const ps::GameState &state) {
  ps::GameState result;
  result.config = state.config;
  result.ball = reflect_point(state.config, state.ball);
  result.to_move = state.to_move;
  result.status = state.status;
  for (const ps::Point point : state.path) {
    result.path.push_back(reflect_point(state.config, point));
  }
  for (const ps::Segment &segment : state.used_segments) {
    result.used_segments.insert(ps::Segment{
        reflect_point(state.config, segment.a),
        reflect_point(state.config, segment.b)});
  }
  for (const auto &[point, count] : state.visit_count) {
    result.visit_count.emplace(reflect_point(state.config, point), count);
  }
  return result;
}

std::string feature_id(const std::vector<std::uint16_t> &active) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const std::uint16_t value : active) {
    hash ^= static_cast<std::uint8_t>(value & 0xffU);
    hash *= 1099511628211ULL;
    hash ^= static_cast<std::uint8_t>(value >> 8U);
    hash *= 1099511628211ULL;
  }
  std::ostringstream rendered;
  rendered << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16)
           << hash;
  return rendered.str();
}

std::string sampled_action(const native::SearchResult &search,
                           float temperature, SplitMix64 &random) {
  if (temperature <= 0.0F || search.root_actions.size() < 2) {
    return search.encoded;
  }
  std::vector<double> scores;
  scores.reserve(search.root_actions.size());
  double maximum = -std::numeric_limits<double>::infinity();
  for (const native::RootActionStat &action : search.root_actions) {
    const double score = static_cast<double>(action.value) +
                         std::log(static_cast<double>(
                             std::max<std::uint32_t>(1, action.visits)));
    scores.push_back(score);
    maximum = std::max(maximum, score);
  }
  double total = 0.0;
  for (double &score : scores) {
    score = std::exp(std::clamp((score - maximum) / temperature, -80.0, 0.0));
    total += score;
  }
  double selected = random.unit() * total;
  for (std::size_t index = 0; index < scores.size(); ++index) {
    selected -= scores[index];
    if (selected <= 0.0) return search.root_actions[index].encoded;
  }
  return search.root_actions.back().encoded;
}

std::string procedural_turn(ps::GameState &state, SplitMix64 &random) {
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("cannot extend a terminal procedural opening");
  }
  const ps::Player mover = state.to_move;
  std::string encoded;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    if (legal.empty()) {
      throw std::logic_error("nonterminal opening position has no legal move");
    }
    const ps::Move move = legal[random.index(legal.size())];
    encoded.push_back(native::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (encoded.empty()) {
    throw std::logic_error("procedural opening emitted an empty complete turn");
  }
  return encoded;
}

Opening make_opening(const ps::RulesConfig &rules, std::uint64_t game_seed,
                     std::size_t depth) {
  constexpr std::size_t kMaximumRetries = 4'096;
  for (std::size_t retry = 0; retry < kMaximumRetries; ++retry) {
    const std::uint64_t opening_seed =
        game_seed ^ 0xa4093822299f31d0ULL ^
        (static_cast<std::uint64_t>(retry) * 0xd1b54a32d192ed03ULL);
    SplitMix64 random(opening_seed);
    Opening result{ps::make_initial_state(rules), depth, opening_seed, retry, {}};
    bool valid = true;
    for (std::size_t turn = 0; turn < depth; ++turn) {
      if (ps::is_terminal(result.state)) {
        valid = false;
        break;
      }
      if (!result.transcript.empty()) result.transcript.push_back('/');
      result.transcript += procedural_turn(result.state, random);
    }
    if (valid && !ps::is_terminal(result.state)) return result;
  }
  throw std::runtime_error("could not construct a nonterminal procedural opening");
}

native::SearchConfig search_config(std::uint64_t seed,
                                   const native::QuantizedModel *model,
                                   std::uint64_t work) {
  native::SearchConfig config;
  config.max_expansions = native::kMaximumExpansions;
  config.max_tree_nodes = static_cast<std::size_t>(
      std::min<std::uint64_t>(work, native::kMaximumTreeNodes));
  config.shuffle_seed = seed;
  config.model = model;
  return config;
}

std::vector<Sample> select_evenly(std::vector<Sample> samples,
                                  std::size_t maximum) {
  if (samples.size() <= maximum) return samples;
  if (maximum == 1) return {std::move(samples.front())};
  std::vector<Sample> result;
  result.reserve(maximum);
  for (std::size_t index = 0; index < maximum; ++index) {
    const std::size_t source = index * (samples.size() - 1) / (maximum - 1);
    result.push_back(std::move(samples[source]));
  }
  return result;
}

void write_record(std::ostream &output, const Arguments &arguments,
                  std::size_t game, std::uint64_t game_seed,
                  int winner, std::size_t complete_turns,
                  const std::vector<Sample> &samples,
                  const LoadedModel &player_one,
                  const LoadedModel &player_two,
                  const Opening &opening,
                  std::string_view transcript,
                  const AggregateSearchStats &search_stats) {
  output << "{\"schema\":\"papersoccer.jacek-native-game/v1\",";
  output << "\"feature_schema\":\""
         << papersoccer::jacek_native_model::kFeatureSchema << "\",";
  output << "\"rules\":{\"width\":8,\"height\":10,"
            "\"goal_rule\":\"own-goals-allowed\","
            "\"blocked_rule\":\"mover-loses\"},";
  output << "\"generator\":{"
            "\"schema\":\"jacek-native-complete-turn-bfm/v1\","
            "\"action\":\"complete-turn\",\"max_actions\":250,"
            "\"deque_schedule\":\"nine-lifo-one-fifo\","
         << "\"search_work\":" << arguments.work << ','
         << "\"work_unit\":\"maximum-tree-nodes\","
         << "\"sampling_temperature\":" << arguments.temperature << ','
         << "\"temperature_turns\":" << arguments.temperature_turns << ','
         << "\"temperature_schedule\":\"" << kTemperatureSchedule << "\","
         << "\"opening_schema\":\"" << kOpeningSchema << "\","
         << "\"opening_depth\":" << opening.depth << ','
         << "\"opening_seed\":\"" << opening.seed << "\","
         << "\"opening_retry\":" << opening.retry << ','
         << "\"opening_transcript\":\"" << opening.transcript << "\","
         << "\"value_target\":\"mover-relative-final-outcome\","
         << "\"checkpoint_color_schedule\":"
            "\"swap-player-checkpoints-on-odd-games\","
         << "\"producer_sha256\":\"" << arguments.producer_sha256 << "\","
         << "\"build_provenance_sha256\":\""
         << arguments.build_provenance_sha256 << "\","
         << "\"models\":{";
  const auto write_model = [&](std::string_view name, const LoadedModel &loaded) {
    output << '\"' << name << "\":{\"model_sha256\":\""
           << (loaded.model ? loaded.model_sha256
                            : std::string(papersoccer::jacek_native_model::kModelSha256))
           << "\",\"packed_sha256\":\""
           << (loaded.model ? loaded.packed_sha256
                            : std::string(papersoccer::jacek_native_model::kPackedSha256))
           << "\",\"artifact_sha256\":\""
           << (loaded.artifact_sha256.empty()
                   ? std::string(papersoccer::jacek_native_model::kModelSha256)
                   : loaded.artifact_sha256)
           << "\"}";
  };
  write_model("player_one", player_one);
  output << ',';
  write_model("player_two", player_two);
  output << "},\"search_stats\":{"
         << "\"searches\":" << search_stats.searches << ','
         << "\"expansions\":" << search_stats.expansions << ','
         << "\"child_evaluations\":" << search_stats.child_evaluations << ','
         << "\"completed_actions\":" << search_stats.completed_actions << ','
         << "\"partial_paths\":" << search_stats.partial_paths << ','
         << "\"tactical_proof_paths\":"
         << search_stats.tactical_proof_paths << ','
         << "\"generator_truncations\":"
         << search_stats.generator_truncations << ','
         << "\"tactical_classes_found\":"
         << search_stats.tactical_classes_found << ','
         << "\"tactical_proof_truncations\":"
         << search_stats.tactical_proof_truncations << ','
         << "\"tree_cap_searches\":" << search_stats.tree_cap_searches << ','
         << "\"expansion_cap_searches\":"
         << search_stats.expansion_cap_searches << "}},";
  output << "\"seed\":\"" << game_seed << "\",\"game\":" << game
         << ",\"shard_index\":" << arguments.shard_index
         << ",\"shard_count\":" << arguments.shard_count
         << ",\"split_group\":\"native:" << game_seed << ':' << game
         << "\",\"winner\":" << winner
         << ",\"complete_turns\":" << complete_turns
         << ",\"transcript_schema\":"
            "\"complete-turn-directions-slash/v1\","
         << "\"transcript\":\"" << transcript << "\",\"samples\":[";
  for (std::size_t sample_index = 0; sample_index < samples.size();
       ++sample_index) {
    const Sample &sample = samples[sample_index];
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
    if (arguments.reanalysis_work != 0) {
      const native::QuantizedModel *sample_model =
          sample.player == ps::Player::One ? player_one.model.get()
                                           : player_two.model.get();
      const native::SearchResult reanalysis_one = native::choose_complete_turn(
          sample.state,
          search_config(game_seed ^ (sample.turn * 0x9e3779b97f4a7c15ULL),
                        sample_model, arguments.reanalysis_work));
      const std::uint64_t verification_work =
          std::min<std::uint64_t>(100'000, arguments.reanalysis_work * 2U);
      const native::SearchResult reanalysis_two = native::choose_complete_turn(
          sample.state,
          search_config(game_seed ^ (sample.turn * 0x9e3779b97f4a7c15ULL),
                        sample_model, verification_work));
      const bool truncated =
          reanalysis_one.stats.deadline_reached ||
          reanalysis_one.stats.generator_truncations != 0 ||
          reanalysis_one.stats.tree_cap_reached ||
          reanalysis_one.stats.expansion_cap_reached ||
          reanalysis_two.stats.deadline_reached ||
          reanalysis_two.stats.generator_truncations != 0 ||
          reanalysis_two.stats.tree_cap_reached ||
          reanalysis_two.stats.expansion_cap_reached;
      const float first_value = std::clamp(reanalysis_one.value, -1.0F, 1.0F);
      const float second_value = std::clamp(reanalysis_two.value, -1.0F, 1.0F);
      const bool action_stable =
          reanalysis_one.encoded == reanalysis_two.encoded;
      const float value_delta = std::abs(first_value - second_value);
      const bool exact = reanalysis_two.solved &&
                         reanalysis_two.solved_winner.has_value();
      const float selected_value = exact
          ? (*reanalysis_two.solved_winner == sample.player ? 1.0F : -1.0F)
          : second_value;
      const bool stable = exact ||
                          (!truncated && action_stable && value_delta <= 0.05F);
      output << ",\"reanalysis\":{\"value\":"
             << selected_value << ",\"work\":" << arguments.reanalysis_work
             << ",\"verification_work\":" << verification_work
             << ",\"truncated\":" << (truncated ? "true" : "false")
             << ",\"action_stable\":" << (action_stable ? "true" : "false")
             << ",\"value_delta\":" << value_delta
             << ",\"stable\":" << (stable ? "true" : "false")
             << ",\"exact\":" << (exact ? "true" : "false") << '}';
    }
    output << '}';
  }
  output << "]}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    if (std::filesystem::exists(arguments.output)) {
      throw std::runtime_error("refusing to overwrite existing corpus output");
    }
    std::ofstream output(arguments.output);
    if (!output) throw std::runtime_error("could not open corpus output");
    const LoadedModel base_player_one = load_checkpoint(
        arguments.player_one_checkpoint, arguments.player_one_artifact_sha256);
    const LoadedModel base_player_two = load_checkpoint(
        arguments.player_two_checkpoint, arguments.player_two_artifact_sha256);
    std::size_t completed = 0;
    for (std::size_t game = arguments.shard_index; game < arguments.games;
         game += arguments.shard_count) {
      const std::uint64_t game_seed =
          arguments.seed + game * 0x9e3779b97f4a7c15ULL;
      const bool swap_models = (game & 1U) != 0U;
      const LoadedModel &player_one =
          swap_models ? base_player_two : base_player_one;
      const LoadedModel &player_two =
          swap_models ? base_player_one : base_player_two;
      const std::size_t opening_depth =
          arguments.opening_depths[game % arguments.opening_depths.size()];
      SplitMix64 random(game_seed ^ 0xd1b54a32d192ed03ULL);
      ps::RulesConfig rules;
      rules.width = 8;
      rules.height = 10;
      rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
      rules.blocked_rule = ps::BlockedRule::MoverLoses;
      const Opening opening = make_opening(rules, game_seed, opening_depth);
      ps::GameState state = opening.state;
      std::string transcript = opening.transcript;
      std::vector<Sample> samples;
      AggregateSearchStats search_stats;
      std::size_t turn = opening_depth;
      for (; turn < arguments.maximum_complete_turns && !ps::is_terminal(state);
           ++turn) {
        std::vector<std::uint16_t> active = active_features(state);
        std::vector<std::uint16_t> reflected =
            active_features(reflected_state(state));
        samples.push_back(Sample{
            turn, state.to_move, state, active, feature_id(active),
            reflected, feature_id(reflected)});
        const native::QuantizedModel *turn_model =
            state.to_move == ps::Player::One ? player_one.model.get()
                                              : player_two.model.get();
        const native::SearchResult search = native::choose_complete_turn(
            state, search_config(game_seed ^ random.next(), turn_model,
                                 arguments.work));
        search_stats.observe(search.stats);
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
      const int winner = *ps::winner(state) == ps::Player::One ? 0 : 1;
      const std::size_t complete_turns = turn;
      samples = select_evenly(std::move(samples), arguments.samples_per_game);
      write_record(output, arguments, game, game_seed, winner,
                   complete_turns, samples, player_one, player_two,
                   opening, transcript, search_stats);
      ++completed;
      if (completed % 10 == 0) {
        std::cerr << "completed " << completed << " shard games\n";
      }
    }
    std::cerr << "completed " << completed << " shard games\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Jacek-native self-play: " << error.what() << '\n';
    return 1;
  }
}
