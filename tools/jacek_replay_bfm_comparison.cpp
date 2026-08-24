#include "jacek_replay_bfm_gate_internal.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"
#include "mcts_internal.hpp"
#include "papersoccer/bot.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;
namespace gate = papersoccer::jacek_replay_gate;

namespace {

using Clock = std::chrono::steady_clock;

#ifndef PAPERSOCCER_JACEK_REPLAY_RANK4_CONTROL_SHA256
#error "Rank-4 control SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_CONTROL_SHA256
#error "neural PUCT control SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_JACEK_NN_CONTROL_SHA256
#error "jacek_nn control SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_RANK4_ENGINE_SHA256
#error "Rank-4 engine SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ENGINE_SHA256
#error "neural PUCT engine SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ENGINE_SHA256
#error "jacek_nn engine SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_RANK4_ADAPTER_SHA256
#error "Rank-4 adapter SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ADAPTER_SHA256
#error "neural PUCT adapter SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ADAPTER_SHA256
#error "jacek_nn adapter SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_SHA256
#error "jacek_nn source closure SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_SHA256
#error "shared control core SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_SHA256
#error "candidate source closure SHA-256 must be bound by the build"
#endif
#ifndef PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_SHA256
#error "comparison source SHA-256 must be bound by the build"
#endif

enum class Opponent {
  Rank4,
  NeuralPuct,
  JacekNn,
  JacekReplay,
  Both,
  Rank4AndJacekNn
};

std::string_view opponent_name(Opponent opponent) {
  switch (opponent) {
    case Opponent::Rank4:
      return "rank4";
    case Opponent::NeuralPuct:
      return "neural-puct";
    case Opponent::JacekNn:
      return "jacek-nn";
    case Opponent::JacekReplay:
      return "jacek-replay";
    case Opponent::Both:
      return "both";
    case Opponent::Rank4AndJacekNn:
      return "rank4-jacek-nn";
  }
  throw std::logic_error("unknown opponent selection");
}

struct Options {
  std::string model_path;
  std::string control_model_path;
  std::string output{"-"};
  std::string bank_path;
  std::string generate_bank_path;
  std::string tuning_receipt_path;
  std::string baseline_receipt_path;
  std::string bank_classification{"development"};
  Opponent opponent{Opponent::Both};
  std::size_t pairs{6};
  std::size_t pair_offset{};
  std::size_t opening_plies{12};
  std::size_t max_turns{320};
  std::uint64_t seed{0x4A5242464D5631ULL};
  std::uint32_t time_ms{980};
  std::uint64_t control_work{3'000'000};
  std::size_t tree_nodes{1'000'000};
  std::size_t control_tree_nodes{100'000};
  std::size_t max_actions{250};
  std::size_t max_partial_paths{50'000};
  double exploration{0.95};
  double fpu{0.5};
  std::uint64_t opening_bank_seed{};
  std::size_t opening_bank_minimum_physical_plies{};
  std::string comparison_executable_path{};
  std::string comparison_executable_sha256{};
  std::string model_sha256{};
  std::string control_model_sha256{};
  std::string opening_bank_sha256{};
  std::vector<std::string> opening_state_identities{};
};

struct Opening {
  std::string id;
  std::string transcript;
  ps::GameState state;
};

struct Work {
  std::uint64_t decisions{};
  std::uint64_t work{};
  std::uint64_t neural_evaluations{};
  std::uint64_t deadlines{};
  std::uint64_t caps{};
  std::uint64_t replay_corrections{};
  double milliseconds{};
  double maximum_ms{};
  std::vector<double> samples_ms{};
};

struct Game {
  std::string opening;
  std::string opponent;
  int candidate_player{};
  int winner{-1};
  std::size_t turns{};
  bool illegal{};
  std::string error;
  std::string transcript;
  Work candidate;
  Work control;
};

template <typename UInt>
UInt parse_unsigned(std::string_view raw, std::string_view option,
                    bool allow_zero = false) {
  UInt value{};
  const auto [end, error] =
      std::from_chars(raw.data(), raw.data() + raw.size(), value);
  if (raw.empty() || error != std::errc{} || end != raw.data() + raw.size() ||
      (!allow_zero && value == 0)) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

std::string required_value(int &index, int argc, char **argv,
                           std::string_view option) {
  if (++index >= argc) {
    throw std::invalid_argument("missing value for " + std::string(option));
  }
  return argv[index];
}

double parse_double(std::string_view raw, std::string_view option) {
  std::string owned(raw);
  char *end = nullptr;
  const double value = std::strtod(owned.c_str(), &end);
  if (owned.empty() || end != owned.c_str() + owned.size() ||
      !std::isfinite(value)) {
    throw std::invalid_argument(std::string(option) + " requires a number");
  }
  return value;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view option(argv[index]);
    if (option == "--help") {
      std::cout
          << "usage: papersoccer_jacek_replay_comparison --model PATH "
             "[--control-model PATH] "
             "[--output PATH|-] "
             "[--bank opening.tsv] "
             "[--generate-bank opening.tsv] "
             "[--tuning-receipt tuning.json] "
             "[--baseline-receipt baseline.json] "
             "[--bank-classification development|final] "
             "[--opponent rank4|neural-puct|jacek-nn|jacek-replay|both|"
             "rank4-jacek-nn] [--pairs N] "
             "[--pair-offset N] "
             "[--opening-plies N] [--max-turns N] [--seed N] "
             "[--time-ms N] [--control-work N] [--tree-nodes N] "
             "[--control-tree-nodes N] [--max-actions N] "
             "[--max-partial-paths N] [--exploration X] [--fpu X]\n";
      std::exit(0);
    }
    const std::string value = required_value(index, argc, argv, option);
    if (option == "--model") options.model_path = value;
    else if (option == "--control-model") options.control_model_path = value;
    else if (option == "--output") options.output = value;
    else if (option == "--bank") options.bank_path = value;
    else if (option == "--generate-bank") options.generate_bank_path = value;
    else if (option == "--tuning-receipt") options.tuning_receipt_path = value;
    else if (option == "--baseline-receipt") {
      options.baseline_receipt_path = value;
    }
    else if (option == "--bank-classification") {
      if (value != "development" && value != "final") {
        throw std::invalid_argument("invalid bank classification");
      }
      options.bank_classification = value;
    }
    else if (option == "--opponent") {
      if (value == "rank4") options.opponent = Opponent::Rank4;
      else if (value == "neural-puct") options.opponent = Opponent::NeuralPuct;
      else if (value == "jacek-nn") options.opponent = Opponent::JacekNn;
      else if (value == "jacek-replay") {
        options.opponent = Opponent::JacekReplay;
      }
      else if (value == "both") options.opponent = Opponent::Both;
      else if (value == "rank4-jacek-nn") {
        options.opponent = Opponent::Rank4AndJacekNn;
      }
      else throw std::invalid_argument("invalid opponent");
    } else if (option == "--pairs") {
      options.pairs = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--pair-offset") {
      options.pair_offset =
          parse_unsigned<std::size_t>(value, option, true);
    } else if (option == "--opening-plies") {
      options.opening_plies =
          parse_unsigned<std::size_t>(value, option, true);
    } else if (option == "--max-turns") {
      options.max_turns = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--seed") {
      options.seed = parse_unsigned<std::uint64_t>(value, option, true);
    } else if (option == "--time-ms") {
      options.time_ms = parse_unsigned<std::uint32_t>(value, option);
    } else if (option == "--control-work") {
      options.control_work = parse_unsigned<std::uint64_t>(value, option);
    } else if (option == "--tree-nodes") {
      options.tree_nodes = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--control-tree-nodes") {
      options.control_tree_nodes = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--max-actions") {
      options.max_actions = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--max-partial-paths") {
      options.max_partial_paths = parse_unsigned<std::size_t>(value, option);
    } else if (option == "--exploration") {
      options.exploration = parse_double(value, option);
    } else if (option == "--fpu") {
      options.fpu = parse_double(value, option);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(option));
    }
  }
  if (options.model_path.empty() && options.generate_bank_path.empty()) {
    throw std::invalid_argument("--model is required");
  }
  if (!options.bank_path.empty() && !options.generate_bank_path.empty()) {
    throw std::invalid_argument("--bank and --generate-bank are exclusive");
  }
  if (options.opponent == Opponent::JacekReplay &&
      options.control_model_path.empty()) {
    throw std::invalid_argument(
        "--control-model is required for the jacek-replay opponent");
  }
  if (options.opponent != Opponent::JacekReplay &&
      !options.control_model_path.empty()) {
    throw std::invalid_argument(
        "--control-model requires the jacek-replay opponent");
  }
  if (options.bank_path.empty() && options.pair_offset != 0) {
    throw std::invalid_argument("--pair-offset requires --bank");
  }
  return options;
}

ps::RulesConfig contest_rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed, ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

char encode_opening_direction(ps::Point from, ps::Point to) {
  static constexpr std::array<ps::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
  }};
  const ps::Point delta{to.x - from.x, to.y - from.y};
  for (std::size_t index = 0; index < deltas.size(); ++index) {
    if (delta == deltas[index]) return static_cast<char>('0' + index);
  }
  throw std::logic_error("random opening contains a non-neighbour move");
}

Opening make_opening(std::size_t index, const Options &options) {
  for (std::uint64_t attempt = 0; attempt < 1024; ++attempt) {
    const std::uint64_t seed = options.seed + index * 1'000'003ULL + attempt;
    ps::RandomBot random(seed);
    ps::GameState state = ps::make_initial_state(contest_rules());
    std::string transcript;
    std::size_t physical_plies = 0;
    while (physical_plies < options.opening_plies && !ps::is_terminal(state)) {
      const ps::Player mover = state.to_move;
      std::string action;
      do {
        const ps::Move move = random.choose_move(state);
        action.push_back(encode_opening_direction(state.ball, move.to));
        state = ps::apply_move(state, move);
        ++physical_plies;
      } while (!ps::is_terminal(state) && state.to_move == mover);
      if (!transcript.empty()) transcript.push_back('/');
      transcript += action;
    }
    if (!ps::is_terminal(state)) {
      return {"generated-" + std::to_string(index) + "-" +
                  std::to_string(seed),
              std::move(transcript),
              std::move(state)};
    }
  }
  throw std::runtime_error("could not generate a nonterminal opening");
}

ps::Move decode_direction(ps::Point from, char direction) {
  static constexpr std::array<ps::Point, 8> deltas{{
      {0, -1}, {1, -1}, {1, 0}, {1, 1},
      {0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
  }};
  if (direction < '0' || direction > '7') {
    throw std::invalid_argument("opening contains an invalid direction");
  }
  const ps::Point delta = deltas[static_cast<std::size_t>(direction - '0')];
  return ps::Move{{from.x + delta.x, from.y + delta.y}};
}

void apply_encoded_turn(ps::GameState &state, std::string_view action) {
  if (action.empty()) {
    throw std::invalid_argument("opening contains an empty turn");
  }
  ps::GameState next = state;
  const ps::Player mover = next.to_move;
  for (const char direction : action) {
    if (ps::is_terminal(next) || next.to_move != mover) {
      throw std::invalid_argument("opening turn continues after its boundary");
    }
    next = ps::apply_move(next, decode_direction(next.ball, direction));
  }
  if (!ps::is_terminal(next) && next.to_move == mover) {
    throw std::invalid_argument("opening turn stops during a rebound");
  }
  state = std::move(next);
}

Opening opening_from_transcript(std::string id, std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(contest_rules());
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    apply_encoded_turn(state, transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) break;
    begin = separator + 1U;
  }
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("opening transcript is terminal");
  }
  return Opening{std::move(id), std::string(transcript), std::move(state)};
}

std::string state_identity(const ps::GameState &state) {
  const auto key_for = [](const ps::GameState &value) {
    auto topology =
        std::make_shared<ps::detail::SearchTopology>(value.config);
    ps::detail::SearchPosition position(topology, value);
    return position.position_key();
  };
  const auto transformed = [](const ps::GameState &source, bool rotate,
                              bool reflect) {
    const auto point = [=](ps::Point value) {
      if (rotate) value = {8 - value.x, 12 - value.y};
      if (reflect) value = {8 - value.x, value.y};
      return value;
    };
    ps::GameState result;
    result.config = source.config;
    result.ball = point(source.ball);
    result.to_move = rotate ? ps::opponent(source.to_move) : source.to_move;
    result.status = source.status;
    if (rotate && result.status == ps::Status::WonByOne) {
      result.status = ps::Status::WonByTwo;
    } else if (rotate && result.status == ps::Status::WonByTwo) {
      result.status = ps::Status::WonByOne;
    }
    for (const ps::Point value : source.path) result.path.push_back(point(value));
    for (const ps::Segment &segment : source.used_segments) {
      result.used_segments.insert(
          ps::Segment{point(segment.a), point(segment.b)});
    }
    for (const auto &[value, visits] : source.visit_count) {
      result.visit_count[point(value)] = visits;
    }
    return result;
  };
  std::array<ps::detail::PositionKey, 4> keys{{
      key_for(state),
      key_for(transformed(state, false, true)),
      key_for(transformed(state, true, false)),
      key_for(transformed(state, true, true)),
  }};
  const auto less = [](const ps::detail::PositionKey &left,
                       const ps::detail::PositionKey &right) {
    return left.first < right.first ||
           (left.first == right.first && left.second < right.second);
  };
  const ps::detail::PositionKey key = *std::min_element(
      keys.begin(), keys.end(), less);
  return std::to_string(key.first) + ":" + std::to_string(key.second);
}

std::vector<Opening> load_bank(
    const std::string &path, std::string_view classification,
    std::uint64_t &bank_seed, std::size_t &minimum_physical_plies) {
  std::ifstream input(path);
  if (!input) throw std::invalid_argument("could not open opening bank");
  std::vector<std::string> lines;
  std::string line;
  while (std::getline(input, line)) lines.push_back(line);
  const std::string classification_line =
      "# classification=" + std::string(classification);
  if (lines.size() < 7U ||
      lines[0] != "# papersoccer.jacek-replay-bfm-opening-bank.v1" ||
      lines[1] != "# rules=8x10;own-goals-allowed;mover-loses" ||
      lines[2] != classification_line ||
      !lines[3].starts_with("# seed=") || lines[3].size() == 7U ||
      !lines[4].starts_with("# minimum-physical-plies=") ||
      lines[4].size() == 25U ||
      lines[5] != "opening_id\ttranscript\tstate_identity") {
    throw std::invalid_argument("opening bank metadata contract is invalid");
  }
  bank_seed = parse_unsigned<std::uint64_t>(
      std::string_view(lines[3]).substr(7U), "opening bank seed", true);
  minimum_physical_plies = parse_unsigned<std::size_t>(
      std::string_view(lines[4]).substr(25U),
      "opening bank minimum physical plies", true);
  std::vector<Opening> openings;
  for (std::size_t line_index = 6U; line_index < lines.size(); ++line_index) {
    line = lines[line_index];
    if (line.empty() || line.front() == '#') {
      throw std::invalid_argument("opening bank has unexpected trailing metadata");
    }
    const std::size_t first = line.find('\t');
    const std::size_t second =
        first == std::string::npos ? std::string::npos : line.find('\t', first + 1U);
    if (first == std::string::npos || second == std::string::npos ||
        line.find('\t', second + 1U) != std::string::npos || first == 0U ||
        second + 1U == line.size()) {
      throw std::invalid_argument(
          "opening bank row must be id<TAB>transcript<TAB>state identity");
    }
    Opening opening = opening_from_transcript(
        line.substr(0, first),
        std::string_view(line).substr(first + 1U, second - first - 1U));
    if (state_identity(opening.state) != line.substr(second + 1U)) {
      throw std::invalid_argument("opening bank state identity is stale");
    }
    openings.push_back(std::move(opening));
  }
  if (openings.empty()) throw std::invalid_argument("opening bank is empty");
  std::sort(openings.begin(), openings.end(),
            [](const Opening &left, const Opening &right) {
              return left.id < right.id;
            });
  for (std::size_t index = 1; index < openings.size(); ++index) {
    if (openings[index - 1].id == openings[index].id) {
      throw std::invalid_argument("opening bank contains duplicate ids");
    }
  }
  std::set<std::string> transcripts;
  std::set<std::string> states;
  for (const Opening &opening : openings) {
    if (!transcripts.insert(opening.transcript).second) {
      throw std::invalid_argument("opening bank contains duplicate transcripts");
    }
    if (!states.insert(state_identity(opening.state)).second) {
      throw std::invalid_argument("opening bank contains duplicate states");
    }
  }
  return openings;
}

void generate_bank(const Options &options) {
  std::ofstream output(options.generate_bank_path,
                       std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("could not open generated opening bank");
  }
  output << "# papersoccer.jacek-replay-bfm-opening-bank.v1\n"
         << "# rules=8x10;own-goals-allowed;mover-loses\n"
         << "# classification=" << options.bank_classification << "\n"
         << "# seed=" << options.seed << "\n"
         << "# minimum-physical-plies=" << options.opening_plies << "\n"
         << "opening_id\ttranscript\tstate_identity\n";
  std::set<std::string> transcripts;
  std::set<std::string> states;
  std::size_t candidate_index = 0;
  for (std::size_t pair = 0; pair < options.pairs; ++pair) {
    Opening opening = make_opening(candidate_index++, options);
    while (!transcripts.insert(opening.transcript).second ||
           !states.insert(state_identity(opening.state)).second) {
      if (candidate_index > options.pairs * 1024U) {
        throw std::runtime_error("could not generate unique opening transcripts");
      }
      opening = make_opening(candidate_index++, options);
    }
    output << opening.id << '\t' << opening.transcript << '\t'
           << state_identity(opening.state) << '\n';
  }
  if (!output) {
    throw std::runtime_error("could not write complete generated opening bank");
  }
}

void add_work(Work &target, std::uint64_t work, std::uint64_t evaluations,
              bool deadline, bool cap, double milliseconds,
              bool replay_correction = false) {
  ++target.decisions;
  target.work += work;
  target.neural_evaluations += evaluations;
  target.deadlines += deadline ? 1U : 0U;
  target.caps += cap ? 1U : 0U;
  target.replay_corrections += replay_correction ? 1U : 0U;
  target.milliseconds += milliseconds;
  target.maximum_ms = std::max(target.maximum_ms, milliseconds);
  target.samples_ms.push_back(milliseconds);
}

std::string apply_complete_action(ps::GameState &state,
                                  const std::vector<ps::Move> &action) {
  if (action.empty()) throw std::logic_error("engine returned an empty action");
  const ps::Player mover = state.to_move;
  std::string encoded;
  encoded.reserve(action.size());
  for (const ps::Move move : action) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("engine returned an overlong action");
    }
    encoded.push_back(encode_opening_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("engine stopped during a mandatory rebound");
  }
  return encoded;
}

std::string choose_candidate(ps::JacekReplayBfmBot &bot,
                             ps::GameState &state, Work &work) {
  const ps::Player mover = state.to_move;
  const auto started = Clock::now();
  const ps::Move first = bot.choose_move(state);
  const ps::JacekReplayBfmSearchStats stats = bot.last_search_stats();
  std::string encoded;
  encoded.push_back(encode_opening_direction(state.ball, first.to));
  state = ps::apply_move(state, first);
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const ps::Move move = bot.choose_move(state);
    encoded.push_back(encode_opening_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  const double elapsed =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  add_work(work, stats.expansions + stats.partial_paths,
           stats.neural_evaluations, stats.deadline_reached,
           stats.tree_cap_reached, elapsed);
  return encoded;
}

std::string choose_control(Opponent opponent,
                           const gate::ControlConfig &config,
                           ps::GameState &state,
                           std::string_view transcript, Work &work) {
  const auto started = Clock::now();
  gate::ControlDecision decision;
  if (opponent == Opponent::Rank4) {
    decision = gate::choose_rank4_control(state, config);
  } else if (opponent == Opponent::NeuralPuct) {
    decision = gate::choose_neural_puct_control(state, config);
  } else if (opponent == Opponent::JacekNn) {
    decision = gate::choose_jacek_nn_control(state, transcript, config);
  } else {
    throw std::logic_error("aggregate opponent cannot play a game");
  }
  const std::string encoded = apply_complete_action(state, decision.action);
  const double elapsed =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  add_work(work, decision.work, decision.neural_evaluations,
           decision.deadline_reached, decision.cap_reached, elapsed,
           decision.replay_correction);
  return encoded;
}

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) transcript.push_back('/');
  transcript.append(action);
}

Game play(const Opening &opening, Opponent opponent, int candidate_player,
          const Options &options) {
  Game result;
  result.opening = opening.id;
  result.opponent = opponent_name(opponent);
  result.candidate_player = candidate_player;
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  ps::JacekReplayBfmConfig candidate_config;
  candidate_config.model_path = options.model_path;
  candidate_config.seed = options.seed ^
                          (static_cast<std::uint64_t>(candidate_player) << 63U);
  candidate_config.max_time_ms = options.time_ms;
  candidate_config.max_tree_nodes = options.tree_nodes;
  candidate_config.max_actions = options.max_actions;
  candidate_config.max_partial_paths = options.max_partial_paths;
  candidate_config.exploration = options.exploration;
  candidate_config.fpu = options.fpu;
  ps::JacekReplayBfmBot candidate(candidate_config);
  if (candidate.model_sha256() != options.model_sha256) {
    throw std::runtime_error("candidate model changed during comparison");
  }
  std::unique_ptr<ps::JacekReplayBfmBot> replay_control;
  if (opponent == Opponent::JacekReplay) {
    ps::JacekReplayBfmConfig reference_config = candidate_config;
    reference_config.model_path = options.control_model_path;
    reference_config.seed =
        options.seed ^
        (static_cast<std::uint64_t>(1 - candidate_player) << 63U);
    replay_control =
        std::make_unique<ps::JacekReplayBfmBot>(reference_config);
    if (replay_control->model_sha256() != options.control_model_sha256) {
      throw std::runtime_error("control model changed during comparison");
    }
  }
  gate::ControlConfig control;
  control.time_ms = options.time_ms;
  control.work_cap = options.control_work;
  control.tree_cap = options.control_tree_nodes;

  try {
    while (!ps::is_terminal(state) && result.turns < options.max_turns) {
      if (player_id(state.to_move) == candidate_player) {
        append_turn(transcript,
                    choose_candidate(candidate, state, result.candidate));
      } else {
        if (replay_control) {
          append_turn(transcript,
                      choose_candidate(*replay_control, state,
                                       result.control));
        } else {
          append_turn(transcript,
                      choose_control(opponent, control, state, transcript,
                                     result.control));
        }
      }
      ++result.turns;
    }
    if (const std::optional<ps::Player> winner = ps::winner(state)) {
      result.winner = player_id(*winner);
    }
  } catch (const std::exception &error) {
    result.illegal = true;
    result.error = error.what();
    result.winner = 1 - player_id(state.to_move);
  }
  result.transcript = std::move(transcript);
  return result;
}

std::string json_string(std::string_view value) {
  std::string output{"\""};
  for (const char character : value) {
    if (character == '\"' || character == '\\') output.push_back('\\');
    output.push_back(character);
  }
  output.push_back('\"');
  return output;
}

std::string sha256_file(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::invalid_argument("could not hash opening bank");
  const std::streampos end = input.tellg();
  if (end < 0) throw std::invalid_argument("opening bank size is invalid");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input && !bytes.empty()) {
    throw std::invalid_argument("could not read complete opening bank");
  }
  return ps::detail::replay_bfm_sha256_hex(bytes);
}

void write_work(std::ostream &out, const Work &work) {
  std::vector<double> sorted = work.samples_ms;
  std::sort(sorted.begin(), sorted.end());
  const double p99 = sorted.empty()
                         ? 0.0
                         : sorted[std::min(
                               sorted.size() - 1U,
                               static_cast<std::size_t>(
                                   std::ceil(0.99 * sorted.size())) - 1U)];
  out << "{\"decisions\":" << work.decisions << ",\"work\":" << work.work
      << ",\"neural_evaluations\":" << work.neural_evaluations
      << ",\"deadlines\":" << work.deadlines << ",\"caps\":" << work.caps
      << ",\"replay_corrections\":" << work.replay_corrections
      << ",\"total_ms\":" << work.milliseconds
      << ",\"p99_ms\":" << p99
      << ",\"max_ms\":" << work.maximum_ms << '}';
}

void write_samples(std::ostream &out, const std::vector<double> &samples) {
  out << '[';
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) out << ',';
    out << samples[index];
  }
  out << ']';
}

void write_output(const Options &options, const std::vector<Game> &games) {
  int wins = 0;
  int losses = 0;
  int unfinished = 0;
  int illegal = 0;
  std::array<int, 2> color_games{};
  std::array<int, 2> color_wins{};
  Work candidate;
  Work control;
  for (const Game &game : games) {
    ++color_games[game.candidate_player];
    if (game.winner < 0) ++unfinished;
    else if (game.winner == game.candidate_player) {
      ++wins;
      ++color_wins[game.candidate_player];
    } else ++losses;
    illegal += game.illegal ? 1 : 0;
    candidate.decisions += game.candidate.decisions;
    candidate.work += game.candidate.work;
    candidate.neural_evaluations += game.candidate.neural_evaluations;
    candidate.deadlines += game.candidate.deadlines;
    candidate.caps += game.candidate.caps;
    candidate.replay_corrections += game.candidate.replay_corrections;
    candidate.milliseconds += game.candidate.milliseconds;
    candidate.maximum_ms = std::max(candidate.maximum_ms,
                                    game.candidate.maximum_ms);
    candidate.samples_ms.insert(candidate.samples_ms.end(),
                                game.candidate.samples_ms.begin(),
                                game.candidate.samples_ms.end());
    control.decisions += game.control.decisions;
    control.work += game.control.work;
    control.neural_evaluations += game.control.neural_evaluations;
    control.deadlines += game.control.deadlines;
    control.caps += game.control.caps;
    control.replay_corrections += game.control.replay_corrections;
    control.milliseconds += game.control.milliseconds;
    control.maximum_ms = std::max(control.maximum_ms, game.control.maximum_ms);
    control.samples_ms.insert(control.samples_ms.end(),
                              game.control.samples_ms.begin(),
                              game.control.samples_ms.end());
  }
  std::cout << std::setprecision(17)
            << "{\"schema\":\"papersoccer.jacek-replay-bfm-comparison.v1\","
            << "\"model\":" << json_string(options.model_path)
            << ",\"model_sha256\":";
  std::cout << json_string(options.model_sha256)
            << ",\"configuration\":{\"pairs\":" << options.pairs
            << ",\"pair_offset\":" << options.pair_offset
            << ",\"opening_source\":"
            << json_string(options.bank_path.empty() ? "generated"
                                                     : options.bank_path)
            << ",\"opening_bank_sha256\":";
  if (options.bank_path.empty()) {
    std::cout << "null";
  } else {
    std::cout << json_string(options.opening_bank_sha256);
  }
  std::cout << ",\"opening_state_identities\":[";
  for (std::size_t index = 0;
       index < options.opening_state_identities.size(); ++index) {
    if (index != 0) std::cout << ',';
    std::cout << json_string(options.opening_state_identities[index]);
  }
  std::cout << ']';
  std::cout << ",\"opening_bank_classification\":"
            << json_string(options.bank_classification);
  std::cout << ",\"tuning_receipt_path\":";
  if (options.tuning_receipt_path.empty()) {
    std::cout << "null,\"tuning_receipt_sha256\":null";
  } else {
    std::cout << json_string(options.tuning_receipt_path)
              << ",\"tuning_receipt_sha256\":"
              << json_string(sha256_file(options.tuning_receipt_path));
  }
  std::cout << ",\"baseline_receipt_path\":";
  if (options.baseline_receipt_path.empty()) {
    std::cout << "null,\"baseline_receipt_sha256\":null";
  } else {
    std::cout << json_string(options.baseline_receipt_path)
              << ",\"baseline_receipt_sha256\":"
              << json_string(sha256_file(options.baseline_receipt_path));
  }
  std::cout << ",\"control_model_path\":";
  if (options.control_model_path.empty()) {
    std::cout << "null,\"control_model_sha256\":null";
  } else {
    std::cout << json_string(options.control_model_path)
              << ",\"control_model_sha256\":"
              << json_string(options.control_model_sha256);
  }
  std::cout << ",\"rank4_control_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_RANK4_CONTROL_SHA256)
            << ",\"rank4_engine_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_RANK4_ENGINE_SHA256)
            << ",\"neural_puct_control_sha256\":"
            << json_string(
                   PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_CONTROL_SHA256)
            << ",\"neural_puct_engine_sha256\":"
            << json_string(
                   PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ENGINE_SHA256)
            << ",\"jacek_nn_control_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_CONTROL_SHA256)
            << ",\"jacek_nn_engine_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ENGINE_SHA256)
            << ",\"rank4_adapter_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_RANK4_ADAPTER_SHA256)
            << ",\"neural_puct_adapter_sha256\":"
            << json_string(
                   PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ADAPTER_SHA256)
            << ",\"jacek_nn_adapter_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ADAPTER_SHA256)
            << ",\"jacek_nn_source_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_SHA256)
            << ",\"shared_core_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_SHA256)
            << ",\"candidate_source_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_SHA256)
            << ",\"comparison_source_sha256\":"
            << json_string(PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_SHA256)
            << ",\"comparison_executable_path\":"
            << json_string(options.comparison_executable_path)
            << ",\"comparison_executable_sha256\":"
            << json_string(options.comparison_executable_sha256)
            << ",\"opening_plies\":" << options.opening_plies
            << ",\"opening_bank_seed\":" << options.opening_bank_seed
            << ",\"opening_bank_minimum_physical_plies\":"
            << options.opening_bank_minimum_physical_plies
            << ",\"seed\":" << options.seed
            << ",\"time_ms\":" << options.time_ms
            << ",\"max_turns\":" << options.max_turns
            << ",\"opponent\":"
            << json_string(opponent_name(options.opponent))
            << ",\"candidate_tree_nodes\":" << options.tree_nodes
            << ",\"max_actions\":" << options.max_actions
            << ",\"max_partial_paths\":" << options.max_partial_paths
            << ",\"exploration\":" << options.exploration
            << ",\"fpu\":" << options.fpu
            << ",\"control_tree_nodes\":" << options.control_tree_nodes
            << ",\"control_work\":" << options.control_work
            << ",\"single_thread\":true},\"summary\":{\"games\":"
            << games.size() << ",\"wins\":" << wins
            << ",\"losses\":" << losses
            << ",\"unfinished\":" << unfinished
            << ",\"illegal\":" << illegal
            << ",\"colors\":[{\"games\":" << color_games[0]
            << ",\"wins\":" << color_wins[0] << "},{\"games\":"
            << color_games[1] << ",\"wins\":" << color_wins[1]
            << "}],\"candidate\":";
  write_work(std::cout, candidate);
  std::cout << ",\"control\":";
  write_work(std::cout, control);
  std::cout << "},\"results\":[";
  for (std::size_t index = 0; index < games.size(); ++index) {
    const Game &game = games[index];
    if (index != 0) std::cout << ',';
    std::cout << "{\"opening\":" << json_string(game.opening)
              << ",\"opponent\":" << json_string(game.opponent)
              << ",\"candidate_player\":" << game.candidate_player
              << ",\"winner\":";
    if (game.winner < 0) std::cout << "null";
    else std::cout << game.winner;
    std::cout << ",\"turns\":" << game.turns
              << ",\"illegal\":" << (game.illegal ? "true" : "false");
    if (game.illegal) std::cout << ",\"error\":" << json_string(game.error);
    std::cout << ",\"candidate_ms\":";
    write_samples(std::cout, game.candidate.samples_ms);
    std::cout << ",\"control_ms\":";
    write_samples(std::cout, game.control.samples_ms);
    std::cout << ",\"candidate_work\":";
    write_work(std::cout, game.candidate);
    std::cout << ",\"control_work\":";
    write_work(std::cout, game.control);
    std::cout << ",\"transcript\":" << json_string(game.transcript);
    std::cout << '}';
  }
  std::cout << "]}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    Options options = parse_options(argc, argv);
    options.comparison_executable_path =
        std::filesystem::absolute(argv[0]).lexically_normal().string();
    options.comparison_executable_sha256 =
        sha256_file(options.comparison_executable_path);
    if (!options.generate_bank_path.empty()) {
      generate_bank(options);
      return 0;
    }
    options.model_sha256 = sha256_file(options.model_path);
    if (!options.control_model_path.empty()) {
      options.control_model_sha256 =
          sha256_file(options.control_model_path);
    }
    if (!options.bank_path.empty()) {
      options.opening_bank_sha256 = sha256_file(options.bank_path);
    }
    std::ofstream output;
    std::streambuf *original_output = nullptr;
    if (options.output != "-") {
      output.open(options.output, std::ios::binary | std::ios::trunc);
      if (!output) {
        throw std::runtime_error("could not open comparison output");
      }
      original_output = std::cout.rdbuf(output.rdbuf());
    }
    std::vector<Opponent> opponents;
    if (options.opponent == Opponent::Rank4 ||
        options.opponent == Opponent::Both ||
        options.opponent == Opponent::Rank4AndJacekNn) {
      opponents.push_back(Opponent::Rank4);
    }
    if (options.opponent == Opponent::NeuralPuct ||
        options.opponent == Opponent::Both) {
      opponents.push_back(Opponent::NeuralPuct);
    }
    if (options.opponent == Opponent::JacekNn ||
        options.opponent == Opponent::Rank4AndJacekNn) {
      opponents.push_back(Opponent::JacekNn);
    }
    if (options.opponent == Opponent::JacekReplay) {
      opponents.push_back(Opponent::JacekReplay);
    }
    std::vector<Game> games;
    const std::vector<Opening> bank = options.bank_path.empty()
                                          ? std::vector<Opening>{}
                                          : load_bank(options.bank_path,
                                                      options.bank_classification,
                                                      options.opening_bank_seed,
                                                      options.opening_bank_minimum_physical_plies);
    if (!options.bank_path.empty() &&
        sha256_file(options.bank_path) != options.opening_bank_sha256) {
      throw std::runtime_error("opening bank changed while it was loaded");
    }
    if (!bank.empty() &&
        (options.pair_offset > bank.size() ||
         options.pairs > bank.size() - options.pair_offset)) {
      throw std::invalid_argument(
          "opening bank has fewer requested rows than --pair-offset + --pairs");
    }
    options.opening_state_identities.reserve(options.pairs);
    std::set<std::string> comparison_states;
    for (std::size_t pair = 0; pair < options.pairs; ++pair) {
      const Opening opening =
          bank.empty() ? make_opening(pair, options)
                       : bank[options.pair_offset + pair];
      const std::string identity = state_identity(opening.state);
      if (!comparison_states.insert(identity).second) {
        throw std::invalid_argument(
            "comparison openings repeat a canonical state");
      }
      options.opening_state_identities.push_back(identity);
      for (const Opponent opponent : opponents) {
        games.push_back(play(opening, opponent, 0, options));
        games.push_back(play(opening, opponent, 1, options));
      }
    }
    if (sha256_file(options.model_path) != options.model_sha256) {
      throw std::runtime_error("candidate model changed during comparison");
    }
    if (!options.control_model_path.empty() &&
        sha256_file(options.control_model_path) !=
            options.control_model_sha256) {
      throw std::runtime_error("control model changed during comparison");
    }
    if (!options.bank_path.empty() &&
        sha256_file(options.bank_path) != options.opening_bank_sha256) {
      throw std::runtime_error("opening bank changed during comparison");
    }
    write_output(options, games);
    if (original_output != nullptr) {
      std::cout.flush();
      std::cout.rdbuf(original_output);
      if (!output) {
        throw std::runtime_error("could not write complete comparison output");
      }
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek replay comparison: " << error.what() << '\n';
    return 2;
  }
}
