#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 jacek_replay_continuation_teacher
#include "../submissions/codingame/bots/rank_4/bot.cpp"
#undef turn_action_v2

#if defined(__GNUC__) && !defined(__clang__)
namespace papersoccer::jacek_replay_continuation_jacek_nn {
namespace replay_book {
using namespace ::papersoccer::jacek_replay_continuation_teacher::replay_book;
}
namespace replay_value_model {
using namespace
    ::papersoccer::jacek_replay_continuation_teacher::replay_value_model;
}
}  // namespace papersoccer::jacek_replay_continuation_jacek_nn
#endif

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 jacek_replay_continuation_jacek_nn
#include "../submissions/codingame/bots/jacek_nn/bot.cpp"
#undef turn_action_v2

#include "papersoccer/bot.hpp"
#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"
#include "jacek_replay_continuations_internal.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef PAPERSOCCER_JACEK_SELFSEARCH_CONTINUATION_SOURCE_SHA256
#error "continuation source closure SHA-256 must be provided by CMake"
#endif
#ifndef PAPERSOCCER_JACEK_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256
#error "Rank-4 actor source closure SHA-256 must be provided by CMake"
#endif
#ifndef PAPERSOCCER_JACEK_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256
#error "jacek_nn actor source closure SHA-256 must be provided by CMake"
#endif

namespace ps = papersoccer;
namespace teacher = papersoccer::jacek_replay_continuation_teacher;
namespace jacek_nn = papersoccer::jacek_replay_continuation_jacek_nn;
namespace continuation = papersoccer::jacek_replay_continuations;

namespace {

constexpr std::size_t kAttemptCapPerRequestedGame = 20U;
constexpr std::string_view kManifestSchema =
    "papersoccer.jacek-replay-continuations-manifest.v1";

struct Options {
  std::string input;
  std::string output;
  std::string manifest;
  std::string model;
  std::string runner_up_model;
  std::string selfsearch_plan;
  std::string campaign_id;
  std::size_t games{10'000};
  int round{};
  std::uint64_t seed{0x4A5242464D5631ULL};
  std::uint64_t actor_nodes{16'000};
  std::size_t candidate_tree_nodes{2'000};
  std::uint64_t jacek_nn_nodes{64'000};
  double candidate_exploration{0.95};
  double candidate_fpu{0.5};
  std::size_t maximum_turns{320};
};

struct Root {
  std::string group_id;
  std::vector<std::string> actions;
  std::size_t row_ordinal{};
  std::string transcript_sha256;
};

struct GeneratedGame {
  int winner{};
  std::vector<std::string> transcript;
  std::size_t prefix_turns{};
};

enum class SelfActorMode {
  IncumbentSelfplay,
  IncumbentPlayerOneRank4,
  IncumbentPlayerTwoRank4,
  IncumbentPlayerOneJacekNn,
  IncumbentPlayerTwoJacekNn,
  IncumbentPlayerOneRunnerUp,
  IncumbentPlayerTwoRunnerUp,
  StudentSelfplay,
  StudentPlayerOneRank4,
  StudentPlayerTwoRank4,
  StudentPlayerOneJacekNn,
  StudentPlayerTwoJacekNn,
  StudentPlayerOnePriorIncumbent,
  StudentPlayerTwoPriorIncumbent,
};

struct SelfPlanRow {
  std::size_t game_ordinal{};
  SelfActorMode actor_mode{};
  std::uint64_t base_seed{};
};

struct SelfRecord {
  std::string game_id;
  std::size_t row_ordinal{};
  std::size_t game_ordinal{};
  std::size_t attempt_ordinal{};
  std::uint64_t base_seed{};
  std::uint64_t game_seed{};
  SelfActorMode actor_mode{};
  std::size_t root_row_ordinal{};
  std::string root_group_id;
  std::string root_transcript_sha256;
  std::size_t prefix_turns{};
  int winner{};
  std::string transcript;
  std::string transcript_sha256;
};

struct ContinuationRecord {
  std::string continuation_id;
  std::size_t row_ordinal{};
  std::size_t attempt_ordinal{};
  std::uint64_t game_seed{};
  continuation::ActorMode actor_mode{};
  std::size_t root_row_ordinal{};
  std::string root_group_id;
  std::string root_transcript_sha256;
  std::size_t prefix_turns{};
  int winner{};
  std::string transcript;
  std::string transcript_sha256;
};

class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) noexcept : state_(seed) {}
  std::uint64_t next() noexcept {
    std::uint64_t value = (state_ += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }
  std::size_t index(std::size_t bound) noexcept {
    return bound < 2U ? 0U : static_cast<std::size_t>(next() % bound);
  }

 private:
  std::uint64_t state_{};
};

template <typename UInt>
UInt parse_unsigned(std::string_view raw, std::string_view label,
                    bool allow_zero = false) {
  UInt value{};
  const auto [end, error] =
      std::from_chars(raw.data(), raw.data() + raw.size(), value);
  if (raw.empty() || error != std::errc{} || end != raw.data() + raw.size() ||
      (!allow_zero && value == 0)) {
    throw std::invalid_argument(std::string(label) + " requires an integer");
  }
  return value;
}

std::string require_value(int &index, int argc, char **argv,
                          std::string_view option) {
  if (++index >= argc) {
    throw std::invalid_argument("missing value for " + std::string(option));
  }
  return argv[index];
}

double parse_double(std::string_view raw, std::string_view label) {
  const std::string owned(raw);
  char *end = nullptr;
  const double value = std::strtod(owned.c_str(), &end);
  if (owned.empty() || end != owned.c_str() + owned.size() ||
      !std::isfinite(value)) {
    throw std::invalid_argument(std::string(label) + " requires a number");
  }
  return value;
}

bool is_lower_sha256(std::string_view value) {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

bool is_printable_text(std::string_view value) {
  return !value.empty() &&
         std::none_of(value.begin(), value.end(), [](unsigned char character) {
           return character < 0x20U || character == 0x7fU;
         });
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help") {
      std::cout
          << "usage: papersoccer_jacek_replay_continuations --input roots.tsv "
             "--output continuations.tsv --manifest manifest.json "
             "[--games N] [--round 0|1|2] "
             "[--model checkpoint] [--seed N] [--actor-nodes N] "
             "[--candidate-tree-nodes N] [--max-turns N] "
             "[--selfsearch-plan plan.tsv --campaign-id ID "
             "--runner-up-model checkpoint --jacek-nn-nodes N "
             "--candidate-exploration C --candidate-fpu V]\n";
      std::exit(0);
    }
    const std::string value = require_value(index, argc, argv, option);
    if (option == "--input") options.input = value;
    else if (option == "--output") options.output = value;
    else if (option == "--manifest") options.manifest = value;
    else if (option == "--model") options.model = value;
    else if (option == "--runner-up-model") options.runner_up_model = value;
    else if (option == "--selfsearch-plan") options.selfsearch_plan = value;
    else if (option == "--campaign-id") options.campaign_id = value;
    else if (option == "--games") {
      options.games = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--round") {
      options.round = parse_unsigned<int>(value, option, true);
    } else if (option == "--seed") {
      options.seed = parse_unsigned<std::uint64_t>(value, option, true);
    } else if (option == "--actor-nodes") {
      options.actor_nodes = parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--candidate-tree-nodes") {
      options.candidate_tree_nodes =
          parse_unsigned<std::size_t>(value, option);
    } else if (option == "--jacek-nn-nodes") {
      options.jacek_nn_nodes = parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--candidate-exploration") {
      options.candidate_exploration = parse_double(value, option);
    } else if (option == "--candidate-fpu") {
      options.candidate_fpu = parse_double(value, option);
    } else if (option == "--max-turns") {
      options.maximum_turns = parse_unsigned<std::size_t>(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  const bool selfsearch = !options.selfsearch_plan.empty();
  if (options.input.empty() || options.output.empty() ||
      options.manifest.empty() || options.round < 0 || options.round > 2 ||
      (!selfsearch && options.round > 0 && options.model.empty()) ||
      (!selfsearch && options.round == 0 && !options.model.empty()) ||
      (selfsearch && (options.model.empty() || options.runner_up_model.empty() ||
                      !is_printable_text(options.campaign_id))) ||
      (!std::isfinite(options.candidate_exploration) ||
       options.candidate_exploration < 0.0) ||
      (!std::isfinite(options.candidate_fpu) ||
       options.candidate_fpu < -1.0 || options.candidate_fpu > 1.0) ||
      options.candidate_tree_nodes < 2U ||
      options.candidate_tree_nodes > 1'000'000U ||
      options.actor_nodes > teacher::kMaximumNodes ||
      options.jacek_nn_nodes > jacek_nn::kMaximumNodes ||
      options.games >
          std::numeric_limits<std::size_t>::max() /
              kAttemptCapPerRequestedGame) {
    throw std::invalid_argument("invalid continuation configuration");
  }
  const auto normalized = [](const std::string &raw) {
    return std::filesystem::absolute(raw).lexically_normal();
  };
  const std::filesystem::path input = normalized(options.input);
  const std::filesystem::path output = normalized(options.output);
  const std::filesystem::path manifest = normalized(options.manifest);
  if (input == output || input == manifest || output == manifest ||
      (!options.selfsearch_plan.empty() &&
       (normalized(options.selfsearch_plan) == output ||
        normalized(options.selfsearch_plan) == manifest)) ||
      (!options.model.empty() &&
       (normalized(options.model) == output ||
        normalized(options.model) == manifest)) ||
      (!options.runner_up_model.empty() &&
       (normalized(options.runner_up_model) == output ||
        normalized(options.runner_up_model) == manifest))) {
    throw std::invalid_argument("continuation paths must be distinct");
  }
  return options;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
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

std::string sha256(std::string_view bytes) {
  return ps::detail::replay_bfm_sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
}

std::string read_file(const std::string &path, std::string_view label) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::invalid_argument("could not open " + std::string(label));
  }
  const std::streampos end = input.tellg();
  if (end < 0) {
    throw std::invalid_argument(std::string(label) + " size is invalid");
  }
  std::string bytes(static_cast<std::size_t>(end), '\0');
  input.seekg(0, std::ios::beg);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input && !bytes.empty()) {
    throw std::invalid_argument("could not read complete " +
                                std::string(label));
  }
  return bytes;
}

std::string file_sha256(const std::string &path, std::string_view label) {
  return sha256(read_file(path, label));
}

std::vector<Root> load_roots(std::string_view bytes) {
  std::istringstream input{std::string(bytes)};
  std::vector<Root> roots;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#' ||
        line == "group_id\tsource\twinner\ttranscript") {
      continue;
    }
    const std::vector<std::string_view> fields = split(line, '\t');
    if (fields.size() != 4U || fields[0].empty() || fields[3].empty()) {
      throw std::invalid_argument("invalid replay roots TSV row");
    }
    Root root;
    root.group_id = fields[0];
    root.row_ordinal = roots.size();
    root.transcript_sha256 = sha256(fields[3]);
    for (const std::string_view action : split(fields[3], '/')) {
      if (action.empty()) throw std::invalid_argument("empty replay action");
      root.actions.emplace_back(action);
    }
    roots.push_back(std::move(root));
  }
  if (roots.empty()) throw std::invalid_argument("replay roots TSV is empty");
  return roots;
}

std::string_view self_actor_mode_name(SelfActorMode mode) noexcept {
  switch (mode) {
    case SelfActorMode::IncumbentSelfplay:
      return "incumbent-selfplay";
    case SelfActorMode::IncumbentPlayerOneRank4:
      return "incumbent-p1-vs-rank4";
    case SelfActorMode::IncumbentPlayerTwoRank4:
      return "incumbent-p2-vs-rank4";
    case SelfActorMode::IncumbentPlayerOneJacekNn:
      return "incumbent-p1-vs-jacek-nn";
    case SelfActorMode::IncumbentPlayerTwoJacekNn:
      return "incumbent-p2-vs-jacek-nn";
    case SelfActorMode::IncumbentPlayerOneRunnerUp:
      return "incumbent-p1-vs-runner-up";
    case SelfActorMode::IncumbentPlayerTwoRunnerUp:
      return "incumbent-p2-vs-runner-up";
    case SelfActorMode::StudentSelfplay:
      return "student-selfplay";
    case SelfActorMode::StudentPlayerOneRank4:
      return "student-p1-vs-rank4";
    case SelfActorMode::StudentPlayerTwoRank4:
      return "student-p2-vs-rank4";
    case SelfActorMode::StudentPlayerOneJacekNn:
      return "student-p1-vs-jacek-nn";
    case SelfActorMode::StudentPlayerTwoJacekNn:
      return "student-p2-vs-jacek-nn";
    case SelfActorMode::StudentPlayerOnePriorIncumbent:
      return "student-p1-vs-prior-incumbent";
    case SelfActorMode::StudentPlayerTwoPriorIncumbent:
      return "student-p2-vs-prior-incumbent";
  }
  return "invalid";
}

SelfActorMode parse_self_actor_mode(std::string_view value) {
  constexpr std::array<SelfActorMode, 14> modes{{
      SelfActorMode::IncumbentSelfplay,
      SelfActorMode::IncumbentPlayerOneRank4,
      SelfActorMode::IncumbentPlayerTwoRank4,
      SelfActorMode::IncumbentPlayerOneJacekNn,
      SelfActorMode::IncumbentPlayerTwoJacekNn,
      SelfActorMode::IncumbentPlayerOneRunnerUp,
      SelfActorMode::IncumbentPlayerTwoRunnerUp,
      SelfActorMode::StudentSelfplay,
      SelfActorMode::StudentPlayerOneRank4,
      SelfActorMode::StudentPlayerTwoRank4,
      SelfActorMode::StudentPlayerOneJacekNn,
      SelfActorMode::StudentPlayerTwoJacekNn,
      SelfActorMode::StudentPlayerOnePriorIncumbent,
      SelfActorMode::StudentPlayerTwoPriorIncumbent,
  }};
  for (const SelfActorMode mode : modes) {
    if (self_actor_mode_name(mode) == value) return mode;
  }
  throw std::invalid_argument("self-search game plan has an unknown actor mode");
}

std::vector<SelfPlanRow> load_selfsearch_plan(const std::string &path) {
  std::istringstream input(read_file(path, "self-search game plan"));
  std::vector<SelfPlanRow> rows;
  std::string line;
  std::set<std::uint64_t> base_seeds;
  bool header = false;
  while (std::getline(input, line)) {
    if (line.empty() || line.front() == '#') continue;
    if (!header) {
      if (line != "game_ordinal\tactor_mode\tbase_seed") {
        throw std::invalid_argument("self-search game plan header is invalid");
      }
      header = true;
      continue;
    }
    const std::vector<std::string_view> fields = split(line, '\t');
    if (fields.size() != 3U) {
      throw std::invalid_argument("self-search game plan row is malformed");
    }
    SelfPlanRow row;
    row.game_ordinal = parse_unsigned<std::size_t>(fields[0], "game ordinal", true);
    row.actor_mode = parse_self_actor_mode(fields[1]);
    row.base_seed = parse_unsigned<std::uint64_t>(fields[2], "base seed", true);
    if (!rows.empty() && row.game_ordinal <= rows.back().game_ordinal) {
      throw std::invalid_argument(
          "self-search game ordinals must be strictly increasing");
    }
    if (!base_seeds.insert(row.base_seed).second) {
      throw std::invalid_argument(
          "self-search game plan contains a duplicate base seed");
    }
    rows.push_back(row);
  }
  if (!header || rows.empty()) {
    throw std::invalid_argument("self-search game plan is empty");
  }
  return rows;
}

ps::RulesConfig contest_rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses};
}

char encode_direction(ps::Point from, ps::Point to) {
  static constexpr std::array<ps::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
  }};
  const ps::Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < deltas.size(); ++index) {
    if (delta == deltas[index]) return static_cast<char>('0' + index);
  }
  throw std::logic_error("actor returned a non-neighbour move");
}

std::string rank4_turn(ps::GameState &state, std::uint64_t nodes) {
  teacher::SearchConfig config;
  config.max_nodes = nodes;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  teacher::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  if (action.empty()) throw std::logic_error("Rank-4 actor returned no action");
  const ps::Player mover = state.to_move;
  std::string encoded;
  for (const ps::Move move : action) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("Rank-4 actor returned an overlong action");
    }
    encoded.push_back(encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("Rank-4 actor returned an incomplete action");
  }
  return encoded;
}

std::string random_turn(ps::GameState &state, SplitMix64 &random) {
  const ps::Player mover = state.to_move;
  std::string encoded;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> moves = ps::legal_moves(state);
    if (moves.empty()) throw std::logic_error("random actor has no move");
    const ps::Move move = moves[random.index(moves.size())];
    encoded.push_back(encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  return encoded;
}

std::string candidate_turn(ps::GameState &state, ps::JacekReplayBfmBot &bot) {
  const ps::Player mover = state.to_move;
  std::string encoded;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const ps::Move move = bot.choose_move(state);
    encoded.push_back(encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  return encoded;
}

std::string jacek_nn_turn(ps::GameState &state, std::string_view transcript,
                          std::uint64_t nodes) {
  const int mover_id = state.to_move == ps::Player::One ? 0 : 1;
  std::string encoded;
  if (jacek_nn::try_replay_correction(state, mover_id, transcript, encoded)) {
    return encoded;
  }
  jacek_nn::SearchConfig config;
  config.max_nodes = nodes;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 50;
  jacek_nn::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> action = search.run();
  if (action.empty()) throw std::logic_error("jacek_nn actor returned no action");
  const ps::Player mover = state.to_move;
  for (const ps::Move move : action) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("jacek_nn actor returned an overlong action");
    }
    encoded.push_back(encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("jacek_nn actor returned an incomplete action");
  }
  return encoded;
}

bool selfplay_mode(SelfActorMode mode) noexcept {
  return mode == SelfActorMode::IncumbentSelfplay ||
         mode == SelfActorMode::StudentSelfplay;
}

bool incumbent_actor(SelfActorMode mode, int player) noexcept {
  if (selfplay_mode(mode)) return true;
  switch (mode) {
    case SelfActorMode::IncumbentPlayerOneRank4:
    case SelfActorMode::IncumbentPlayerOneJacekNn:
    case SelfActorMode::IncumbentPlayerOneRunnerUp:
    case SelfActorMode::StudentPlayerOneRank4:
    case SelfActorMode::StudentPlayerOneJacekNn:
    case SelfActorMode::StudentPlayerOnePriorIncumbent:
      return player == 0;
    case SelfActorMode::IncumbentPlayerTwoRank4:
    case SelfActorMode::IncumbentPlayerTwoJacekNn:
    case SelfActorMode::IncumbentPlayerTwoRunnerUp:
    case SelfActorMode::StudentPlayerTwoRank4:
    case SelfActorMode::StudentPlayerTwoJacekNn:
    case SelfActorMode::StudentPlayerTwoPriorIncumbent:
      return player == 1;
    case SelfActorMode::IncumbentSelfplay:
    case SelfActorMode::StudentSelfplay:
      return true;
  }
  return false;
}

bool runner_up_actor(SelfActorMode mode, int player) noexcept {
  switch (mode) {
    case SelfActorMode::IncumbentPlayerOneRunnerUp:
    case SelfActorMode::StudentPlayerOnePriorIncumbent:
      return player == 1;
    case SelfActorMode::IncumbentPlayerTwoRunnerUp:
    case SelfActorMode::StudentPlayerTwoPriorIncumbent:
      return player == 0;
    default:
      return false;
  }
}

bool rank4_actor(SelfActorMode mode, int player) noexcept {
  switch (mode) {
    case SelfActorMode::IncumbentPlayerOneRank4:
    case SelfActorMode::StudentPlayerOneRank4:
      return player == 1;
    case SelfActorMode::IncumbentPlayerTwoRank4:
    case SelfActorMode::StudentPlayerTwoRank4:
      return player == 0;
    default:
      return false;
  }
}

bool jacek_nn_actor(SelfActorMode mode, int player) noexcept {
  switch (mode) {
    case SelfActorMode::IncumbentPlayerOneJacekNn:
    case SelfActorMode::StudentPlayerOneJacekNn:
      return player == 1;
    case SelfActorMode::IncumbentPlayerTwoJacekNn:
    case SelfActorMode::StudentPlayerTwoJacekNn:
      return player == 0;
    default:
      return false;
  }
}

bool candidate_actor(continuation::ActorMode mode, int player) noexcept {
  return mode == continuation::ActorMode::CandidateSelfplay ||
         (mode == continuation::ActorMode::CandidatePlayerOne && player == 0) ||
         (mode == continuation::ActorMode::CandidatePlayerTwo && player == 1);
}

std::optional<GeneratedGame> generate(
    const Root &root, const Options &options, std::uint64_t game_seed,
    continuation::ActorMode actor_mode) {
  SplitMix64 random(game_seed);
  const std::size_t prefix_length = random.index(root.actions.size());
  ps::GameState state = ps::make_initial_state(contest_rules());
  std::vector<std::string> transcript;
  transcript.reserve(prefix_length + options.maximum_turns);
  for (std::size_t turn = 0; turn < prefix_length; ++turn) {
    teacher::apply_encoded_turn(state, root.actions[turn]);
    transcript.push_back(root.actions[turn]);
  }

  std::array<std::unique_ptr<ps::JacekReplayBfmBot>, 2> candidates{};
  if (actor_mode != continuation::ActorMode::Rank4VsRank4) {
    for (int player = 0; player < 2; ++player) {
      if (!candidate_actor(actor_mode, player)) continue;
      ps::JacekReplayBfmConfig config;
      config.model_path = options.model;
      config.seed = game_seed ^ (static_cast<std::uint64_t>(player) << 63U);
      config.max_time_ms = 60'000;
      config.max_tree_nodes = options.candidate_tree_nodes;
      config.max_actions = 250;
      config.max_partial_paths = 5'000;
      candidates[player] =
          std::make_unique<ps::JacekReplayBfmBot>(std::move(config));
    }
  }

  for (std::size_t continuation = 0;
       continuation < options.maximum_turns && !ps::is_terminal(state);
       ++continuation) {
    const int player = state.to_move == ps::Player::One ? 0 : 1;
    const bool explore = continuation < 8U && random.index(100U) < 15U;
    if (explore) {
      transcript.push_back(random_turn(state, random));
    } else if (candidate_actor(actor_mode, player)) {
      transcript.push_back(candidate_turn(state, *candidates[player]));
    } else {
      transcript.push_back(rank4_turn(state, options.actor_nodes));
    }
  }
  const std::optional<ps::Player> winning = ps::winner(state);
  if (!winning.has_value()) return std::nullopt;
  return GeneratedGame{*winning == ps::Player::One ? 0 : 1,
                       std::move(transcript), prefix_length};
}

std::string serialize_transcript(
    const std::vector<std::string> &transcript) {
  std::string encoded;
  std::size_t bytes = transcript.empty() ? 0U : transcript.size() - 1U;
  for (const std::string &turn : transcript) bytes += turn.size();
  encoded.reserve(bytes);
  for (std::size_t turn = 0; turn < transcript.size(); ++turn) {
    if (turn != 0U) encoded.push_back('/');
    encoded += transcript[turn];
  }
  return encoded;
}

std::unique_ptr<ps::JacekReplayBfmBot> make_bfm_actor(
    const std::string &model, const Options &options, std::uint64_t seed) {
  ps::JacekReplayBfmConfig config;
  config.model_path = model;
  config.seed = seed;
  config.max_time_ms = 60'000;
  config.max_tree_nodes = options.candidate_tree_nodes;
  config.max_actions = 250;
  config.max_partial_paths = 5'000;
  config.exploration = options.candidate_exploration;
  config.fpu = options.candidate_fpu;
  return std::make_unique<ps::JacekReplayBfmBot>(std::move(config));
}

std::optional<GeneratedGame> generate_selfsearch(
    const Root &root, const Options &options, std::uint64_t game_seed,
    SelfActorMode actor_mode) {
  SplitMix64 random(game_seed);
  const std::size_t prefix_length = random.index(root.actions.size());
  ps::GameState state = ps::make_initial_state(contest_rules());
  std::vector<std::string> transcript;
  transcript.reserve(prefix_length + options.maximum_turns);
  for (std::size_t turn = 0; turn < prefix_length; ++turn) {
    teacher::apply_encoded_turn(state, root.actions[turn]);
    transcript.push_back(root.actions[turn]);
  }

  std::array<std::unique_ptr<ps::JacekReplayBfmBot>, 2> incumbents{};
  std::array<std::unique_ptr<ps::JacekReplayBfmBot>, 2> runners{};
  for (int player = 0; player < 2; ++player) {
    const std::uint64_t player_seed =
        game_seed ^ (static_cast<std::uint64_t>(player) << 63U);
    if (incumbent_actor(actor_mode, player)) {
      incumbents[player] = make_bfm_actor(options.model, options, player_seed);
    }
    if (runner_up_actor(actor_mode, player)) {
      runners[player] =
          make_bfm_actor(options.runner_up_model, options, player_seed);
    }
  }

  for (std::size_t continuation = 0;
       continuation < options.maximum_turns && !ps::is_terminal(state);
       ++continuation) {
    const int player = state.to_move == ps::Player::One ? 0 : 1;
    const bool explore = continuation < 8U && random.index(100U) < 15U;
    if (explore) {
      transcript.push_back(random_turn(state, random));
    } else if (incumbent_actor(actor_mode, player)) {
      transcript.push_back(candidate_turn(state, *incumbents[player]));
    } else if (runner_up_actor(actor_mode, player)) {
      transcript.push_back(candidate_turn(state, *runners[player]));
    } else if (rank4_actor(actor_mode, player)) {
      transcript.push_back(rank4_turn(state, options.actor_nodes));
    } else if (jacek_nn_actor(actor_mode, player)) {
      transcript.push_back(jacek_nn_turn(
          state, serialize_transcript(transcript), options.jacek_nn_nodes));
    } else {
      throw std::logic_error("self-search actor schedule has no mover");
    }
  }
  const std::optional<ps::Player> winning = ps::winner(state);
  if (!winning.has_value()) return std::nullopt;
  return GeneratedGame{*winning == ps::Player::One ? 0 : 1,
                       std::move(transcript), prefix_length};
}

std::vector<continuation::ActorMode> actor_schedule(
    const continuation::ActorQuotas &quotas, std::uint64_t seed) {
  std::vector<continuation::ActorMode> schedule;
  for (std::size_t raw_mode = 0; raw_mode < quotas.size(); ++raw_mode) {
    schedule.insert(schedule.end(), quotas[raw_mode],
                    static_cast<continuation::ActorMode>(raw_mode));
  }
  SplitMix64 random(seed ^ 0x243f6a8885a308d3ULL);
  for (std::size_t remaining = schedule.size(); remaining > 1U; --remaining) {
    std::swap(schedule[remaining - 1U],
              schedule[random.index(remaining)]);
  }
  return schedule;
}

void append_id_field(std::string &material, std::string_view field) {
  material.append(field);
  material.push_back('\0');
}

std::string continuation_id(const Options &options,
                            const ContinuationRecord &record) {
  std::string material;
  append_id_field(material,
                  "papersoccer.jacek-replay-continuation-id.v1");
  append_id_field(material, std::to_string(options.round));
  append_id_field(material, std::to_string(options.seed));
  append_id_field(material, std::to_string(record.row_ordinal));
  append_id_field(material, std::to_string(record.attempt_ordinal));
  append_id_field(material, std::to_string(record.game_seed));
  append_id_field(material,
                  continuation::actor_mode_name(record.actor_mode));
  append_id_field(material, std::to_string(record.root_row_ordinal));
  append_id_field(material, record.root_group_id);
  append_id_field(material, record.root_transcript_sha256);
  append_id_field(material, std::to_string(record.prefix_turns));
  append_id_field(material, record.transcript_sha256);
  return "continuation:" + sha256(material);
}

std::string selfsearch_game_id(const Options &options,
                               const SelfRecord &record) {
  std::string material;
  append_id_field(material, "papersoccer.jacek-selfsearch-game-id.v1");
  append_id_field(material, options.campaign_id);
  append_id_field(material, std::to_string(record.game_ordinal));
  append_id_field(material, std::to_string(record.attempt_ordinal));
  append_id_field(material, std::to_string(record.base_seed));
  append_id_field(material, std::to_string(record.game_seed));
  append_id_field(material, self_actor_mode_name(record.actor_mode));
  append_id_field(material, record.root_group_id);
  append_id_field(material, record.root_transcript_sha256);
  append_id_field(material, std::to_string(record.prefix_turns));
  append_id_field(material, record.transcript_sha256);
  return "selfsearch-game:" + sha256(material);
}

struct GenerationResult {
  std::vector<ContinuationRecord> records;
  continuation::ActorQuotas successful_quotas{};
  std::size_t attempts{};
};

GenerationResult generate_records(const Options &options,
                                  const std::vector<Root> &roots,
                                  const continuation::ActorQuotas &planned) {
  GenerationResult result;
  result.records.reserve(options.games);
  const std::vector<continuation::ActorMode> schedule =
      actor_schedule(planned, options.seed);
  const std::size_t attempt_cap =
      options.games * kAttemptCapPerRequestedGame;
  while (result.records.size() < schedule.size() &&
         result.attempts < attempt_cap) {
    const std::size_t attempt = result.attempts++;
    const std::uint64_t game_seed =
        options.seed + static_cast<std::uint64_t>(attempt) *
                           0x9e3779b97f4a7c15ULL;
    SplitMix64 root_selector(game_seed ^ 0xd1b54a32d192ed03ULL);
    const Root &root = roots[root_selector.index(roots.size())];
    const continuation::ActorMode actor_mode =
        schedule[result.records.size()];
    const std::optional<GeneratedGame> game =
        generate(root, options, game_seed, actor_mode);
    if (!game.has_value()) continue;

    ContinuationRecord record;
    record.row_ordinal = result.records.size();
    record.attempt_ordinal = attempt;
    record.game_seed = game_seed;
    record.actor_mode = actor_mode;
    record.root_row_ordinal = root.row_ordinal;
    record.root_group_id = root.group_id;
    record.root_transcript_sha256 = root.transcript_sha256;
    record.prefix_turns = game->prefix_turns;
    record.winner = game->winner;
    record.transcript = serialize_transcript(game->transcript);
    record.transcript_sha256 = sha256(record.transcript);
    record.continuation_id = continuation_id(options, record);
    ++result.successful_quotas[static_cast<std::size_t>(actor_mode)];
    result.records.push_back(std::move(record));
  }
  if (result.records.size() != options.games ||
      result.successful_quotas != planned) {
    throw std::runtime_error(
        "could not satisfy exact continuation quotas before attempt cap");
  }
  return result;
}

std::vector<SelfRecord> generate_selfsearch_records(
    const Options &options, const std::vector<Root> &roots,
    const std::vector<SelfPlanRow> &plan) {
  std::vector<SelfRecord> records;
  records.reserve(plan.size());
  for (const SelfPlanRow &planned : plan) {
    std::optional<GeneratedGame> completed;
    std::size_t completed_attempt = 0;
    std::uint64_t completed_seed = 0;
    const Root *completed_root = nullptr;
    for (std::size_t attempt = 0; attempt < kAttemptCapPerRequestedGame;
         ++attempt) {
      const std::uint64_t game_seed =
          planned.base_seed + static_cast<std::uint64_t>(attempt) *
                                  0x9e3779b97f4a7c15ULL;
      SplitMix64 root_selector(game_seed ^ 0xd1b54a32d192ed03ULL);
      const Root &root = roots[root_selector.index(roots.size())];
      completed = generate_selfsearch(root, options, game_seed,
                                      planned.actor_mode);
      if (completed.has_value()) {
        completed_attempt = attempt;
        completed_seed = game_seed;
        completed_root = &root;
        break;
      }
    }
    if (!completed.has_value() || completed_root == nullptr) {
      throw std::runtime_error(
          "self-search game plan row exhausted its attempt cap");
    }
    SelfRecord record;
    record.row_ordinal = records.size();
    record.game_ordinal = planned.game_ordinal;
    record.attempt_ordinal = completed_attempt;
    record.base_seed = planned.base_seed;
    record.game_seed = completed_seed;
    record.actor_mode = planned.actor_mode;
    record.root_row_ordinal = completed_root->row_ordinal;
    record.root_group_id = completed_root->group_id;
    record.root_transcript_sha256 = completed_root->transcript_sha256;
    record.prefix_turns = completed->prefix_turns;
    record.winner = completed->winner;
    record.transcript = serialize_transcript(completed->transcript);
    record.transcript_sha256 = sha256(record.transcript);
    record.game_id = selfsearch_game_id(options, record);
    records.push_back(std::move(record));
  }
  return records;
}

std::string json_string(std::string_view value) {
  static constexpr std::string_view hex = "0123456789abcdef";
  std::string output{"\""};
  for (const unsigned char character : value) {
    switch (character) {
      case '\"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (character < 0x20U) {
          output += "\\u00";
          output.push_back(hex[character >> 4U]);
          output.push_back(hex[character & 0x0fU]);
        } else {
          output.push_back(static_cast<char>(character));
        }
    }
  }
  output.push_back('\"');
  return output;
}

void write_quotas(std::ostream &output,
                  const continuation::ActorQuotas &quotas) {
  output << '{';
  for (std::size_t raw_mode = 0; raw_mode < quotas.size(); ++raw_mode) {
    if (raw_mode != 0U) output << ',';
    const auto mode = static_cast<continuation::ActorMode>(raw_mode);
    output << json_string(continuation::actor_mode_name(mode)) << ':'
           << quotas[raw_mode];
  }
  output << '}';
}

std::string make_tsv(const Options &options,
                     const std::vector<ContinuationRecord> &records) {
  std::ostringstream output;
  output << "# papersoccer.jacek-replay-continuations.v1\n"
         << "# round=" << options.round << "\n"
         << "# games=" << options.games << "\n"
         << "# seed=" << options.seed << "\n"
         << "# actor-policy="
         << (options.round == 0
                 ? "rank4-vs-rank4"
                 : "50%-candidate-selfplay+50%-candidate-rank4-balanced")
         << "\n"
         << "# early-exploration-percent=15\n"
         << "group_id\tsource\twinner\ttranscript\n";
  for (const ContinuationRecord &record : records) {
    output << record.root_group_id << "\tcontinuation-round-"
           << options.round << '\t' << record.winner << '\t'
           << record.transcript << '\n';
  }
  if (!output) throw std::runtime_error("could not serialize continuations");
  return output.str();
}

std::string make_selfsearch_tsv(const Options &options,
                                const std::vector<SelfRecord> &records) {
  std::ostringstream output;
  output << "group_id\tsource\twinner\ttranscript\n";
  for (const SelfRecord &record : records) {
    output << record.root_group_id << '\t' << options.campaign_id << '\t'
           << record.winner << '\t' << record.transcript << '\n';
  }
  if (!output) throw std::runtime_error("could not serialize self-search games");
  return output.str();
}

std::string make_selfsearch_manifest(
    const Options &options, const std::vector<SelfPlanRow> &plan,
    const std::vector<SelfRecord> &records, std::string_view roots_sha256,
    std::string_view plan_sha256, std::string_view output_sha256,
    std::string_view incumbent_sha256, std::string_view runner_sha256) {
  if (plan.size() != records.size()) {
    throw std::logic_error("self-search plan/output count mismatch");
  }
  std::ostringstream output;
  output << "{\"schema\":\"papersoccer.jacek-selfsearch-games.v1\","
         << "\"campaign_id\":" << json_string(options.campaign_id) << ','
         << "\"requested_games\":" << plan.size() << ','
         << "\"successful_games\":" << records.size() << ','
         << "\"configuration\":{\"bfm_tree_nodes\":"
         << options.candidate_tree_nodes << ",\"rank4_nodes\":"
         << options.actor_nodes << ",\"jacek_nn_nodes\":"
         << options.jacek_nn_nodes << ",\"exploration\":"
         << options.candidate_exploration << ",\"fpu\":"
         << options.candidate_fpu
         << ",\"early_exploration_percent\":15,"
         << "\"early_exploration_turns\":8,\"maximum_turns\":"
         << options.maximum_turns << ",\"producer_source_sha256\":"
         << json_string(PAPERSOCCER_JACEK_SELFSEARCH_CONTINUATION_SOURCE_SHA256)
         << ",\"rank4_actor_source_sha256\":"
         << json_string(PAPERSOCCER_JACEK_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256)
         << ",\"jacek_nn_actor_source_sha256\":"
         << json_string(
                PAPERSOCCER_JACEK_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256)
         << "},\"bindings\":{\"roots_sha256\":"
         << json_string(roots_sha256) << ",\"plan_sha256\":"
         << json_string(plan_sha256) << ",\"output_sha256\":"
         << json_string(output_sha256) << ",\"incumbent_model_sha256\":"
         << json_string(incumbent_sha256)
         << ",\"runner_up_model_sha256\":"
         << json_string(runner_sha256)
         << ",\"producer_source_sha256\":"
         << json_string(PAPERSOCCER_JACEK_SELFSEARCH_CONTINUATION_SOURCE_SHA256)
         << ",\"rank4_actor_source_sha256\":"
         << json_string(PAPERSOCCER_JACEK_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256)
         << ",\"jacek_nn_actor_source_sha256\":"
         << json_string(
                PAPERSOCCER_JACEK_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256)
         << "},\"rows\":[";
  for (std::size_t index = 0; index < records.size(); ++index) {
    if (index != 0U) output << ',';
    const SelfRecord &record = records[index];
    output << "{\"game_id\":" << json_string(record.game_id)
           << ",\"row_ordinal\":" << record.row_ordinal
           << ",\"game_ordinal\":" << record.game_ordinal
           << ",\"attempt_ordinal\":" << record.attempt_ordinal
           << ",\"base_seed\":" << record.base_seed
           << ",\"game_seed\":" << record.game_seed
           << ",\"actor_mode\":"
           << json_string(self_actor_mode_name(record.actor_mode))
           << ",\"root_group_id\":" << json_string(record.root_group_id)
           << ",\"prefix_turns\":" << record.prefix_turns
           << ",\"root_lineage\":{\"root_row_ordinal\":"
           << record.root_row_ordinal << ",\"root_transcript_sha256\":"
           << json_string(record.root_transcript_sha256)
           << ",\"prefix_turns\":" << record.prefix_turns
           << "},\"winner\":" << record.winner
           << ",\"transcript_sha256\":"
           << json_string(record.transcript_sha256) << '}';
  }
  output << "]}\n";
  if (!output) {
    throw std::runtime_error("could not serialize self-search game manifest");
  }
  return output.str();
}

std::string make_manifest(const Options &options,
                          const continuation::ActorQuotas &planned,
                          const GenerationResult &generated,
                          std::string_view input_sha256,
                          std::string_view output_sha256,
                          const std::optional<std::string> &model_sha256) {
  std::ostringstream output;
  output << '{'
         << "\"schema\":" << json_string(kManifestSchema) << ','
         << "\"tsv_schema\":"
         << json_string("papersoccer.jacek-replay-continuations.v1") << ','
         << "\"round\":" << options.round << ','
         << "\"requested_games\":" << options.games << ','
         << "\"successful_games\":" << generated.records.size() << ','
         << "\"seed\":" << options.seed << ','
         << "\"actor_nodes\":" << options.actor_nodes << ','
         << "\"candidate_tree_nodes\":" << options.candidate_tree_nodes
         << ','
         << "\"maximum_turns\":" << options.maximum_turns << ','
         << "\"attempt_cap_per_requested_game\":"
         << kAttemptCapPerRequestedGame << ','
         << "\"attempts\":" << generated.attempts << ','
         << "\"failed_attempts\":"
         << generated.attempts - generated.records.size() << ','
         << "\"quota_policy\":"
         << json_string(
                options.round == 0
                    ? "all-rank4-vs-rank4"
                    : "largest-remainder-2:1:1-ties-selfplay,p1,p2")
         << ",\"planned_quotas\":";
  write_quotas(output, planned);
  output << ",\"successful_quotas\":";
  write_quotas(output, generated.successful_quotas);
  output << ",\"bindings\":{\"input_sha256\":"
         << json_string(input_sha256) << ",\"output_sha256\":"
         << json_string(output_sha256) << ",\"model_sha256\":";
  if (model_sha256.has_value()) output << json_string(*model_sha256);
  else output << "null";
  output << "},\"rows\":[";
  for (std::size_t index = 0; index < generated.records.size(); ++index) {
    if (index != 0U) output << ',';
    const ContinuationRecord &record = generated.records[index];
    output << "{\"continuation_id\":"
           << json_string(record.continuation_id)
           << ",\"row_ordinal\":" << record.row_ordinal
           << ",\"attempt_ordinal\":" << record.attempt_ordinal
           << ",\"game_seed\":" << record.game_seed
           << ",\"actor_mode\":"
           << json_string(continuation::actor_mode_name(record.actor_mode))
           << ",\"candidate_color\":"
           << json_string(continuation::candidate_color(record.actor_mode))
           << ",\"root_lineage\":{\"root_row_ordinal\":"
           << record.root_row_ordinal << ",\"group_id\":"
           << json_string(record.root_group_id)
           << ",\"root_transcript_sha256\":"
           << json_string(record.root_transcript_sha256)
           << ",\"prefix_turns\":" << record.prefix_turns
           << "},\"transcript_sha256\":"
           << json_string(record.transcript_sha256) << '}';
  }
  output << "]}\n";
  if (!output) {
    throw std::runtime_error("could not serialize continuation manifest");
  }
  return output.str();
}

void atomic_write(const std::string &raw_path, std::string_view bytes) {
  const std::filesystem::path path(raw_path);
  std::filesystem::path temporary = path;
  temporary += ".tmp.";
  temporary += std::to_string(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("could not open temporary output");
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.flush();
    if (!output) {
      output.close();
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
      throw std::runtime_error("could not write complete temporary output");
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw std::runtime_error("could not atomically publish output: " +
                             error.message());
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (!is_lower_sha256(
            PAPERSOCCER_JACEK_SELFSEARCH_CONTINUATION_SOURCE_SHA256) ||
        !is_lower_sha256(
            PAPERSOCCER_JACEK_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256) ||
        !is_lower_sha256(
            PAPERSOCCER_JACEK_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256)) {
      throw std::logic_error("continuation source identity is invalid");
    }
    const Options options = parse_options(argc, argv);
    const std::string input_bytes =
        read_file(options.input, "replay roots TSV");
    const std::string input_digest = sha256(input_bytes);
    const std::optional<std::string> model_digest =
        options.model.empty()
            ? std::nullopt
            : std::optional<std::string>(
                  file_sha256(options.model, "candidate model"));
    const std::vector<Root> roots = load_roots(input_bytes);
    if (!options.selfsearch_plan.empty()) {
      const std::string plan_bytes =
          read_file(options.selfsearch_plan, "self-search game plan");
      const std::string plan_digest = sha256(plan_bytes);
      const std::string incumbent_digest =
          file_sha256(options.model, "incumbent model");
      const std::string runner_digest =
          file_sha256(options.runner_up_model, "runner-up model");
      const std::vector<SelfPlanRow> plan =
          load_selfsearch_plan(options.selfsearch_plan);
      const std::vector<SelfRecord> records =
          generate_selfsearch_records(options, roots, plan);
      if (file_sha256(options.model, "incumbent model") != incumbent_digest ||
          file_sha256(options.runner_up_model, "runner-up model") !=
              runner_digest ||
          file_sha256(options.input, "replay roots TSV") != input_digest ||
          file_sha256(options.selfsearch_plan, "self-search game plan") !=
              plan_digest) {
        throw std::runtime_error(
            "self-search input changed during game generation");
      }
      const std::string tsv = make_selfsearch_tsv(options, records);
      const std::string output_digest = sha256(tsv);
      const std::string manifest = make_selfsearch_manifest(
          options, plan, records, input_digest, plan_digest, output_digest,
          incumbent_digest, runner_digest);
      atomic_write(options.output, tsv);
      atomic_write(options.manifest, manifest);
      return 0;
    }
    const continuation::ActorQuotas planned =
        continuation::planned_quotas(options.round, options.games);
    const GenerationResult generated =
        generate_records(options, roots, planned);
    if (model_digest.has_value() &&
        file_sha256(options.model, "candidate model") != *model_digest) {
      throw std::runtime_error(
          "candidate model changed during continuation generation");
    }
    if (file_sha256(options.input, "replay roots TSV") != input_digest) {
      throw std::runtime_error(
          "replay roots changed during continuation generation");
    }
    const std::string tsv = make_tsv(options, generated.records);
    const std::string output_digest = sha256(tsv);
    const std::string manifest =
        make_manifest(options, planned, generated, input_digest,
                      output_digest, model_digest);
    atomic_write(options.output, tsv);
    atomic_write(options.manifest, manifest);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek replay continuations: " << error.what() << '\n';
    return 2;
  }
}
