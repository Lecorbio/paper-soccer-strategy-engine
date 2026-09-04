#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 compact_rank4_engine
#include "../rank_4/submission.cpp"
#undef turn_action_v2

#define COMPACT_VALUE_BFM_NO_MAIN
#ifndef COMPACT_VALUE_BFM_CANDIDATE_SOURCE
#define COMPACT_VALUE_BFM_CANDIDATE_SOURCE "submission.cpp"
#endif
#include COMPACT_VALUE_BFM_CANDIDATE_SOURCE

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <system_error>

namespace ps = papersoccer;
namespace rank4 = papersoccer::compact_rank4_engine;
namespace cv = compact_value_bfm;

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::string_view kSchema =
    "papersoccer.compact-value-bfm-rank4-gate.v1";
constexpr std::string_view kBankSchema =
    "papersoccer.compact-value-bfm-opening-bank.v1";
constexpr std::string_view kRank4Sha256 =
    "5c7ebbb38e3b08940eb26ca8cd7585dc5cbce5ad949dfd595bfb0eaab1de53c9";
constexpr double kFirstHardMs = 1000.0;
constexpr double kLaterHardMs = 200.0;
constexpr double kFirstHeadroomMs = 900.0;
constexpr double kLaterHeadroomMs = 180.0;

enum class Mode { FixedWork, ActualClock };
enum class Engine { Candidate, Rank4 };

struct Config {
  std::filesystem::path bank;
  std::filesystem::path candidate_source{
      "submissions/codingame/bots/compact_value_bfm/submission.cpp"};
  std::filesystem::path rank4_source{
      "submissions/codingame/bots/rank_4/submission.cpp"};
  std::filesystem::path output;
  std::string expected_bank_sha;
  std::string expected_candidate_sha;
  std::size_t pair_offset{};
  std::size_t pair_count{};
  Mode mode{Mode::FixedWork};
  std::size_t candidate_actions{cv::kMaximumActions};
  std::size_t candidate_root_partials{cv::kRootPartialPaths};
  std::size_t candidate_nonroot_partials{cv::kNonrootPartialPaths};
  std::size_t candidate_nodes{cv::kProductionTreeNodes};
  std::uint64_t candidate_expansions{cv::kMaximumExpansions};
  std::uint64_t candidate_seed{cv::GeneratorConfig{}.shuffle_seed};
  double candidate_c{cv::kExploration};
  double candidate_fpu{cv::kFirstPlayUrgency};
  double candidate_lambda{cv::kFinalVisitWeight};
  std::uint64_t rank4_nodes{rank4::kMaximumNodes};
  int max_turns{320};
  int minimum_wins{-1};
  int minimum_per_color{-1};
  bool self_test{};
};

struct MatchState {
  ps::GameState core;
  cv::State compact;
  std::string transcript;
};

struct Opening {
  std::string id;
  std::string transcript;
  std::size_t physical_plies{};
  std::size_t complete_turns{};
  MatchState state;
};

struct Bank {
  std::string sha256;
  std::uint64_t bytes{};
  std::vector<Opening> openings;
};

struct Invocation {
  Engine engine{Engine::Candidate};
  std::string action;
  std::string failure;
  std::string failure_detail;
  double milliseconds{};
  bool first{};
  bool deadline_reached{};
  bool soft_overrun{};
  bool headroom_failure{};
  bool hard_timeout{};
  std::uint64_t work{};
  std::uint64_t generated_children{};
  std::uint64_t evaluated_children{};
};

struct EngineSummary {
  int decisions{};
  int deadline_stops{};
  int soft_overruns{};
  int headroom_failures{};
  int hard_timeouts{};
  std::uint64_t work{};
  std::uint64_t generated_children{};
  std::uint64_t evaluated_children{};
  double maximum_first_ms{};
  double maximum_later_ms{};
  std::vector<double> times;
};

struct GameResult {
  std::string opening_id;
  std::size_t pair_index{};
  int candidate_player{};
  int winner{-1};
  int turns{};
  std::string failure;
  std::string failure_detail;
  EngineSummary candidate;
  EngineSummary rank4;
};

struct GateSummary {
  int games{};
  int candidate_wins{};
  int rank4_wins{};
  std::array<int, 2> candidate_wins_by_color{};
  int failures{};
  int unfinished{};
  std::map<std::string, int> failure_categories;
  EngineSummary candidate;
  EngineSummary rank4;
  bool passed{};
};

template <typename Integer>
Integer integer(std::string_view text, std::string_view field) {
  Integer value{};
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument(std::string(field) + " is not an integer");
  }
  return value;
}

double real(std::string_view text, std::string_view field) {
  std::size_t consumed = 0;
  const double value = std::stod(std::string(text), &consumed);
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(field) + " is not finite");
  }
  return value;
}

bool valid_sha(std::string_view value) {
  return value.empty() ||
         (value.size() == 64 &&
          std::all_of(value.begin(), value.end(), [](char character) {
            return (character >= '0' && character <= '9') ||
                   (character >= 'a' && character <= 'f');
          }));
}

std::vector<std::uint8_t> read_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::invalid_argument("cannot open " + path.string());
  return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(input), {});
}

std::string file_sha(const std::filesystem::path &path,
                     std::uint64_t *size = nullptr) {
  const std::vector<std::uint8_t> bytes = read_bytes(path);
  if (size != nullptr) *size = bytes.size();
  return cv::sha256_hex(bytes);
}

std::string json_string(std::string_view value) {
  std::string result{"\""};
  for (const unsigned char character : value) {
    switch (character) {
      case '\\': result += "\\\\"; break;
      case '"': result += "\\\""; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (character < 0x20U) {
          std::ostringstream escaped;
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(character);
          result += escaped.str();
        } else {
          result.push_back(static_cast<char>(character));
        }
    }
  }
  result.push_back('"');
  return result;
}

ps::RulesConfig codingame_rules() {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  return rules;
}

int player_id(ps::Player player) noexcept {
  return player == ps::Player::One ? 0 : 1;
}

std::string_view engine_name(Engine engine) noexcept {
  return engine == Engine::Candidate ? "candidate" : "rank4";
}

cv::Action compact_action(std::string_view encoded) {
  cv::Action action;
  if (encoded.empty() || encoded.size() > action.directions.size()) {
    throw std::invalid_argument("malformed complete-turn action length");
  }
  for (const char character : encoded) {
    if (character < '0' || character > '7') {
      throw std::invalid_argument("malformed complete-turn direction");
    }
    action.directions[action.length++] =
        static_cast<std::uint8_t>(character - '0');
  }
  return action;
}

std::pair<ps::Point, ps::Point> compact_edge_points(int edge) {
  const auto &topology = cv::Topology::get();
  for (int vertex = 0; vertex < cv::kVertexCount; ++vertex) {
    for (int index = 0; index < topology.degree(vertex); ++index) {
      const auto arc = topology.arc(vertex, index);
      if (arc.edge == edge) {
        return {{topology.x(vertex), topology.y(vertex)},
                {topology.x(arc.destination), topology.y(arc.destination)}};
      }
    }
  }
  throw std::logic_error("compact edge is absent from topology");
}

void verify_lockstep(const ps::GameState &core, const cv::State &compact) {
  const auto &topology = cv::Topology::get();
  if (core.ball.x != topology.x(compact.ball) ||
      core.ball.y != topology.y(compact.ball) ||
      player_id(core.to_move) != compact.to_move ||
      ps::is_terminal(core) != compact.terminal() ||
      core.used_segments.size() != compact.ply) {
    throw std::runtime_error("core/compact dynamic-state mismatch");
  }
  const std::optional<ps::Player> winner = ps::winner(core);
  if ((winner.has_value() ? player_id(*winner) : -1) != compact.winner) {
    throw std::runtime_error("core/compact winner mismatch");
  }
  std::size_t compact_edges = 0;
  for (int edge = 0; edge < cv::kEdgeCount; ++edge) {
    if (!cv::edge_used(compact, edge)) continue;
    ++compact_edges;
    const auto [first, second] = compact_edge_points(edge);
    if (!core.used_segments.contains(ps::Segment{first, second})) {
      throw std::runtime_error("core/compact used-edge mismatch");
    }
  }
  if (compact_edges != core.used_segments.size()) {
    throw std::runtime_error("core/compact used-edge count mismatch");
  }
  for (int vertex = 0; vertex < cv::kVertexCount; ++vertex) {
    const ps::Point point{topology.x(vertex), topology.y(vertex)};
    const auto found = core.visit_count.find(point);
    const bool core_visited =
        ps::is_boundary_point(core.config, point) ||
        (found != core.visit_count.end() && found->second > 0);
    const bool compact_visited = cv::vertex_visited(compact, vertex) ||
        (compact.ball == vertex &&
         (topology.north_goal(vertex) || topology.south_goal(vertex)));
    if (compact_visited != core_visited) {
      throw std::runtime_error("core/compact visited-vertex mismatch");
    }
  }
}

void append_transcript(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) transcript.push_back('/');
  transcript.append(action);
}

void apply_both(MatchState &state, std::string_view encoded) {
  ps::GameState next_core = state.core;
  cv::State next_compact = state.compact;
  rank4::apply_encoded_turn(next_core, encoded);
  if (!cv::apply_action(next_compact, compact_action(encoded))) {
    throw std::invalid_argument("compact engine rejected core-legal action");
  }
  verify_lockstep(next_core, next_compact);
  state.core = std::move(next_core);
  state.compact = next_compact;
}

std::vector<std::string_view> split(std::string_view value, char separator) {
  std::vector<std::string_view> result;
  std::size_t begin = 0;
  for (;;) {
    const std::size_t end = value.find(separator, begin);
    result.push_back(value.substr(
        begin, end == std::string_view::npos ? value.size() - begin
                                             : end - begin));
    if (end == std::string_view::npos) return result;
    begin = end + 1U;
  }
}

Opening replay_opening(std::string id, std::string transcript) {
  if (id.empty() || transcript.empty()) {
    throw std::invalid_argument("opening id/transcript is empty");
  }
  if (!std::all_of(id.begin(), id.end(), [](char character) {
        return std::isalnum(static_cast<unsigned char>(character)) ||
               character == '_' || character == '-' || character == '.' ||
               character == ':';
      })) {
    throw std::invalid_argument("opening id is not canonical");
  }
  Opening opening;
  opening.id = std::move(id);
  opening.transcript = std::move(transcript);
  opening.state = MatchState{ps::make_initial_state(codingame_rules()),
                             cv::initial_state(), {}};
  for (const std::string_view action : split(opening.transcript, '/')) {
    if (action.empty() || ps::is_terminal(opening.state.core)) {
      throw std::invalid_argument("opening transcript is malformed or terminal");
    }
    opening.physical_plies += action.size();
    ++opening.complete_turns;
    apply_both(opening.state, action);
    append_transcript(opening.state.transcript, action);
  }
  if (opening.physical_plies < 12U || ps::is_terminal(opening.state.core) ||
      opening.state.compact.terminal()) {
    throw std::invalid_argument(
        "opening must be live with at least 12 physical plies");
  }
  verify_lockstep(opening.state.core, opening.state.compact);
  return opening;
}

std::vector<Opening> parse_bank_text(std::string_view text) {
  std::istringstream input{std::string(text)};
  std::string line;
  bool header = false;
  std::set<std::string> ids;
  std::vector<Opening> openings;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') {
      throw std::invalid_argument("opening bank must use LF line endings");
    }
    if (line.empty() || line.starts_with("#")) continue;
    if (!header) {
      if (line != "opening_id\ttranscript") {
        throw std::invalid_argument("opening bank header mismatch");
      }
      header = true;
      continue;
    }
    const std::size_t tab = line.find('\t');
    if (tab == std::string::npos || line.find('\t', tab + 1U) !=
                                        std::string::npos) {
      throw std::invalid_argument("opening bank row must have two fields");
    }
    std::string id = line.substr(0, tab);
    if (!ids.insert(id).second) {
      throw std::invalid_argument("opening bank contains duplicate id");
    }
    openings.push_back(replay_opening(
        std::move(id), line.substr(tab + 1U)));
  }
  if (!header || openings.empty()) {
    throw std::invalid_argument("opening bank is empty");
  }
  return openings;
}

Bank load_bank(const Config &config) {
  const std::vector<std::uint8_t> bytes = read_bytes(config.bank);
  const std::string sha = cv::sha256_hex(bytes);
  if (!config.expected_bank_sha.empty() && sha != config.expected_bank_sha) {
    throw std::invalid_argument("opening bank SHA-256 mismatch");
  }
  Bank bank{sha, bytes.size(), parse_bank_text(std::string_view(
                                  reinterpret_cast<const char *>(bytes.data()),
                                  bytes.size()))};
  if (config.pair_count == 0 ||
      config.pair_offset > bank.openings.size() ||
      config.pair_count > bank.openings.size() - config.pair_offset) {
    throw std::invalid_argument("pair offset/count exceeds opening bank");
  }
  return bank;
}

std::uint32_t clock_budget(Engine engine, bool first) noexcept {
  if (engine == Engine::Candidate) return first ? 800U : 155U;
  return first ? 800U : 165U;
}

void classify_time(Invocation &result, Mode mode) noexcept {
  if (mode != Mode::ActualClock) return;
  const double soft = clock_budget(result.engine, result.first);
  result.soft_overrun = result.milliseconds > soft;
  result.headroom_failure = result.milliseconds >=
      (result.first ? kFirstHeadroomMs : kLaterHeadroomMs);
  result.hard_timeout = result.milliseconds >=
      (result.first ? kFirstHardMs : kLaterHardMs);
  if (result.hard_timeout && result.failure.empty()) {
    result.failure = std::string(engine_name(result.engine)) + "_timeout";
  }
}

cv::SearchConfig candidate_config(const Config &config) {
  cv::SearchConfig result;
  result.max_actions = config.candidate_actions;
  result.root_partial_paths = config.candidate_root_partials;
  result.nonroot_partial_paths = config.candidate_nonroot_partials;
  result.max_tree_nodes = config.candidate_nodes;
  result.max_expansions = config.candidate_expansions;
  result.shuffle_seed = config.candidate_seed;
  result.exploration = config.candidate_c;
  result.fpu = config.candidate_fpu;
  result.final_visit_weight = config.candidate_lambda;
  return result;
}

Invocation invoke_candidate(MatchState &state, bool first,
                            const Config &config) {
  Invocation result;
  result.engine = Engine::Candidate;
  result.first = first;
  Clock::time_point started{};
  bool timing_started = false;
  cv::Action emergency{};
  bool emergency_ready = false;
  try {
    cv::SearchConfig search_config = candidate_config(config);
    emergency = cv::emergency_complete_action(
        state.compact, search_config.shuffle_seed);
    emergency_ready = emergency.length != 0;
    started = Clock::now();
    timing_started = true;
    const auto deadline = config.mode == Mode::ActualClock
        ? started + std::chrono::milliseconds(clock_budget(result.engine, first))
        : Clock::time_point::max();
    const cv::SearchResult decision = cv::search(
        state.compact, deadline, search_config, &emergency);
    result.action = decision.action.text();
    result.deadline_reached = decision.stats.deadline_reached;
    result.work = decision.stats.expansions;
    result.generated_children = decision.stats.generated_children;
    result.evaluated_children = decision.stats.evaluated_children;
    if (result.action.empty()) {
      result.failure = "candidate_malformed";
    } else {
      try {
        apply_both(state, result.action);
      } catch (const std::invalid_argument &error) {
        result.failure = "candidate_illegal";
        result.failure_detail = error.what();
      } catch (const std::runtime_error &error) {
        result.failure = "lockstep_mismatch";
        result.failure_detail = error.what();
      }
    }
  } catch (const std::exception &error) {
    if (emergency_ready) {
      result.action = emergency.text();
      result.deadline_reached = true;
      try {
        apply_both(state, result.action);
      } catch (const std::exception &fallback_error) {
        result.failure = "candidate_exception";
        result.failure_detail = std::string(error.what()) +
            "; emergency fallback: " + fallback_error.what();
      }
    } else {
      result.failure = "candidate_exception";
      result.failure_detail = error.what();
    }
  } catch (...) {
    if (emergency_ready) {
      result.action = emergency.text();
      result.deadline_reached = true;
      try {
        apply_both(state, result.action);
      } catch (...) {
        result.failure = "candidate_exception";
        result.failure_detail = "unknown exception and emergency fallback failed";
      }
    } else {
      result.failure = "candidate_exception";
    }
  }
  if (timing_started) {
    result.milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  classify_time(result, config.mode);
  return result;
}

Invocation invoke_rank4(MatchState &state, bool first, const Config &config) {
  Invocation result;
  result.engine = Engine::Rank4;
  result.first = first;
  Clock::time_point started{};
  bool timing_started = false;
  try {
    started = Clock::now();
    timing_started = true;
    std::string encoded;
    rank4::SearchStats stats;
    ps::GameState correction_probe = state.core;
    if (!rank4::try_replay_correction(
            correction_probe, player_id(state.core.to_move),
            state.transcript, encoded)) {
      rank4::SearchConfig search_config;
      search_config.max_nodes = config.rank4_nodes;
      search_config.replay_value_blend_percent = 15;
      search_config.teacher_residual_weight_percent = 100;
      if (config.mode == Mode::ActualClock) {
        search_config.absolute_deadline =
            started + std::chrono::milliseconds(clock_budget(result.engine, first));
      }
      rank4::CompleteTurnSearch search(state.core, search_config);
      const std::vector<ps::Move> moves = search.run();
      stats = search.stats();
      ps::GameState cursor = state.core;
      const ps::Player mover = cursor.to_move;
      for (const ps::Move move : moves) {
        if (ps::is_terminal(cursor) || cursor.to_move != mover) {
          throw std::logic_error("Rank-4 action overrun");
        }
        encoded.push_back(rank4::encode_direction(cursor.ball, move.to));
        cursor = ps::apply_move(cursor, move);
      }
    }
    result.milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
    result.action = std::move(encoded);
    result.deadline_reached = stats.budget_exhausted;
    result.work = stats.nodes;
    if (result.action.empty()) {
      result.failure = "rank4_malformed";
    } else {
      try {
        apply_both(state, result.action);
      } catch (const std::invalid_argument &error) {
        result.failure = "rank4_illegal";
        result.failure_detail = error.what();
      } catch (const std::runtime_error &error) {
        result.failure = "lockstep_mismatch";
        result.failure_detail = error.what();
      }
    }
  } catch (const std::exception &error) {
    result.failure = "rank4_exception";
    result.failure_detail = error.what();
  } catch (...) {
    result.failure = "rank4_exception";
  }
  if (timing_started && result.milliseconds == 0.0) {
    result.milliseconds =
        std::chrono::duration<double, std::milli>(Clock::now() - started).count();
  }
  classify_time(result, config.mode);
  return result;
}

void add_invocation(EngineSummary &summary, const Invocation &invocation) {
  ++summary.decisions;
  summary.deadline_stops += invocation.deadline_reached;
  summary.soft_overruns += invocation.soft_overrun;
  summary.headroom_failures += invocation.headroom_failure;
  summary.hard_timeouts += invocation.hard_timeout;
  summary.work += invocation.work;
  summary.generated_children += invocation.generated_children;
  summary.evaluated_children += invocation.evaluated_children;
  summary.times.push_back(invocation.milliseconds);
  if (invocation.first) {
    summary.maximum_first_ms =
        std::max(summary.maximum_first_ms, invocation.milliseconds);
  } else {
    summary.maximum_later_ms =
        std::max(summary.maximum_later_ms, invocation.milliseconds);
  }
}

GameResult play(const Opening &opening, std::size_t pair_index,
                int candidate_player, const Config &config) {
  MatchState state = opening.state;
  GameResult result;
  result.opening_id = opening.id;
  result.pair_index = pair_index;
  result.candidate_player = candidate_player;
  result.turns = static_cast<int>(opening.complete_turns);
  std::array<int, 2> responses{};
  while (!ps::is_terminal(state.core) && result.turns < config.max_turns) {
    verify_lockstep(state.core, state.compact);
    const bool candidate_turn = player_id(state.core.to_move) == candidate_player;
    const Engine engine = candidate_turn ? Engine::Candidate : Engine::Rank4;
    int &response_count = responses[engine == Engine::Candidate ? 0U : 1U];
    Invocation invocation = engine == Engine::Candidate
        ? invoke_candidate(state, response_count == 0, config)
        : invoke_rank4(state, response_count == 0, config);
    ++response_count;
    add_invocation(candidate_turn ? result.candidate : result.rank4, invocation);
    if (!invocation.failure.empty()) {
      result.failure = invocation.failure;
      result.failure_detail = invocation.failure_detail;
      return result;
    }
    append_transcript(state.transcript, invocation.action);
    ++result.turns;
  }
  verify_lockstep(state.core, state.compact);
  if (const std::optional<ps::Player> winner = ps::winner(state.core)) {
    result.winner = player_id(*winner);
  } else {
    result.failure = "unfinished";
  }
  return result;
}

void merge_engine(EngineSummary &target, const EngineSummary &source) {
  target.decisions += source.decisions;
  target.deadline_stops += source.deadline_stops;
  target.soft_overruns += source.soft_overruns;
  target.headroom_failures += source.headroom_failures;
  target.hard_timeouts += source.hard_timeouts;
  target.work += source.work;
  target.generated_children += source.generated_children;
  target.evaluated_children += source.evaluated_children;
  target.maximum_first_ms =
      std::max(target.maximum_first_ms, source.maximum_first_ms);
  target.maximum_later_ms =
      std::max(target.maximum_later_ms, source.maximum_later_ms);
  target.times.insert(target.times.end(), source.times.begin(), source.times.end());
}

GateSummary summarize(const std::vector<GameResult> &games,
                      const Config &config) {
  GateSummary summary;
  summary.games = games.size();
  for (const GameResult &game : games) {
    merge_engine(summary.candidate, game.candidate);
    merge_engine(summary.rank4, game.rank4);
    if (!game.failure.empty()) {
      ++summary.failures;
      ++summary.failure_categories[game.failure];
      summary.unfinished += game.failure == "unfinished";
      continue;
    }
    if (game.winner == game.candidate_player) {
      ++summary.candidate_wins;
      ++summary.candidate_wins_by_color[game.candidate_player];
    } else {
      ++summary.rank4_wins;
    }
  }
  summary.passed = summary.failures == 0;
  if (config.minimum_wins >= 0) {
    summary.passed = summary.passed &&
                     summary.candidate_wins >= config.minimum_wins;
  }
  if (config.minimum_per_color >= 0) {
    summary.passed = summary.passed &&
        summary.candidate_wins_by_color[0] >= config.minimum_per_color &&
        summary.candidate_wins_by_color[1] >= config.minimum_per_color;
  }
  return summary;
}

std::string engine_json(const EngineSummary &engine) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "{\"decisions\":" << engine.decisions
         << ",\"deadline_stops\":" << engine.deadline_stops
         << ",\"soft_overruns\":" << engine.soft_overruns
         << ",\"headroom_failures\":" << engine.headroom_failures
         << ",\"hard_timeouts\":" << engine.hard_timeouts
         << ",\"work\":" << engine.work
         << ",\"generated_children\":" << engine.generated_children
         << ",\"evaluated_children\":" << engine.evaluated_children
         << ",\"maximum_first_ms\":" << engine.maximum_first_ms
         << ",\"maximum_later_ms\":" << engine.maximum_later_ms
         << ",\"times_ms\":[";
  for (std::size_t index = 0; index < engine.times.size(); ++index) {
    if (index != 0) output << ',';
    output << engine.times[index];
  }
  output << "]}";
  return output.str();
}

std::string game_json(const GameResult &game) {
  std::ostringstream output;
  output << "{\"opening_id\":" << json_string(game.opening_id)
         << ",\"pair_index\":" << game.pair_index
         << ",\"candidate_player\":" << game.candidate_player
         << ",\"winner\":" << game.winner
         << ",\"turns\":" << game.turns
         << ",\"failure\":"
         << (game.failure.empty() ? "null" : json_string(game.failure))
         << ",\"failure_detail\":"
         << (game.failure_detail.empty() ? "null"
                                         : json_string(game.failure_detail))
         << ",\"candidate\":" << engine_json(game.candidate)
         << ",\"rank4\":" << engine_json(game.rank4) << '}';
  return output.str();
}

std::string render(const Config &config, const Bank &bank,
                   std::string_view candidate_sha,
                   std::uint64_t candidate_bytes,
                   std::uint64_t rank4_bytes,
                   const std::vector<GameResult> &games,
                   const GateSummary &summary) {
  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "{\"schema\":" << json_string(kSchema)
         << ",\"bindings\":{\"candidate_source_sha256\":"
         << json_string(candidate_sha)
         << ",\"candidate_source_bytes\":" << candidate_bytes
         << ",\"candidate_runtime_body_sha256\":"
         << json_string(cv::model::kRuntimeBodySha256)
         << ",\"candidate_payload_sha256\":"
         << json_string(cv::model::kPayloadSha256)
         << ",\"rank4_source_sha256\":" << json_string(kRank4Sha256)
         << ",\"rank4_source_bytes\":" << rank4_bytes
         << ",\"opponent_sha256\":" << json_string(kRank4Sha256)
         << ",\"bank_sha256\":" << json_string(bank.sha256)
         << ",\"bank_bytes\":" << bank.bytes << "}"
         << ",\"config\":{\"mode\":"
         << json_string(config.mode == Mode::FixedWork ? "fixed-work"
                                                       : "actual-clock")
         << ",\"pair_offset\":" << config.pair_offset
         << ",\"pair_count\":" << config.pair_count
         << ",\"candidate_c\":" << config.candidate_c
         << ",\"candidate_fpu\":" << config.candidate_fpu
         << ",\"candidate_lambda\":" << config.candidate_lambda
         << ",\"candidate_actions\":" << config.candidate_actions
         << ",\"candidate_root_partial_paths\":"
         << config.candidate_root_partials
         << ",\"candidate_nonroot_partial_paths\":"
         << config.candidate_nonroot_partials
         << ",\"candidate_nodes\":" << config.candidate_nodes
         << ",\"candidate_expansions\":" << config.candidate_expansions
         << ",\"candidate_shuffle_seed\":" << config.candidate_seed
         << ",\"candidate_clocks_ms\":[800,155]"
         << ",\"rank4_nodes\":" << config.rank4_nodes
         << ",\"rank4_clocks_ms\":[800,165]"
         << ",\"max_turns\":" << config.max_turns
         << ",\"minimum_candidate_wins\":" << config.minimum_wins
         << ",\"minimum_wins_per_color\":" << config.minimum_per_color
         << "}"
         << ",\"games\":[";
  for (std::size_t index = 0; index < games.size(); ++index) {
    if (index != 0) output << ',';
    output << game_json(games[index]);
  }
  output << "]"
         << ",\"result\":{\"games\":" << summary.games
         << ",\"candidate_wins\":" << summary.candidate_wins
         << ",\"rank4_wins\":" << summary.rank4_wins
         << ",\"candidate_wins_player0\":"
         << summary.candidate_wins_by_color[0]
         << ",\"candidate_wins_player1\":"
         << summary.candidate_wins_by_color[1]
         << ",\"failures\":" << summary.failures
         << ",\"unfinished\":" << summary.unfinished
         << ",\"failure_categories\":{";
  bool first = true;
  for (const auto &[category, count] : summary.failure_categories) {
    if (!first) output << ',';
    first = false;
    output << json_string(category) << ':' << count;
  }
  output << "},\"candidate\":" << engine_json(summary.candidate)
         << ",\"rank4\":" << engine_json(summary.rank4)
         << ",\"passed\":" << (summary.passed ? "true" : "false")
         << "}}\n";
  return output.str();
}

Config arguments(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--self-test") {
      config.self_test = true;
      continue;
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + std::string(argument));
    }
    const std::string_view value = argv[++index];
    if (argument == "--bank") config.bank = value;
    else if (argument == "--candidate-source") config.candidate_source = value;
    else if (argument == "--rank4-source") config.rank4_source = value;
    else if (argument == "--output") config.output = value;
    else if (argument == "--expected-bank-sha256") config.expected_bank_sha = value;
    else if (argument == "--expected-candidate-sha256") config.expected_candidate_sha = value;
    else if (argument == "--pair-offset") config.pair_offset = integer<std::size_t>(value, "pair offset");
    else if (argument == "--pair-count") config.pair_count = integer<std::size_t>(value, "pair count");
    else if (argument == "--mode") {
      if (value == "fixed-work") config.mode = Mode::FixedWork;
      else if (value == "actual-clock") config.mode = Mode::ActualClock;
      else throw std::invalid_argument("mode must be fixed-work or actual-clock");
    } else if (argument == "--candidate-c") config.candidate_c = real(value, "candidate C");
    else if (argument == "--candidate-fpu") config.candidate_fpu = real(value, "candidate FPU");
    else if (argument == "--candidate-lambda") config.candidate_lambda = real(value, "candidate lambda");
    else if (argument == "--candidate-actions") config.candidate_actions = integer<std::size_t>(value, "candidate actions");
    else if (argument == "--candidate-root-partial-paths") config.candidate_root_partials = integer<std::size_t>(value, "candidate root partial paths");
    else if (argument == "--candidate-nonroot-partial-paths") config.candidate_nonroot_partials = integer<std::size_t>(value, "candidate nonroot partial paths");
    else if (argument == "--candidate-nodes") config.candidate_nodes = integer<std::size_t>(value, "candidate nodes");
    else if (argument == "--candidate-expansions") config.candidate_expansions = integer<std::uint64_t>(value, "candidate expansions");
    else if (argument == "--candidate-seed") config.candidate_seed = integer<std::uint64_t>(value, "candidate seed");
    else if (argument == "--rank4-nodes") config.rank4_nodes = integer<std::uint64_t>(value, "Rank-4 nodes");
    else if (argument == "--max-turns") config.max_turns = integer<int>(value, "max turns");
    else if (argument == "--minimum-candidate-wins") config.minimum_wins = integer<int>(value, "minimum wins");
    else if (argument == "--minimum-wins-per-color") config.minimum_per_color = integer<int>(value, "minimum wins per color");
    else throw std::invalid_argument("unknown argument " + std::string(argument));
  }
  if (config.self_test) return config;
  if (config.bank.empty() || config.expected_bank_sha.empty() ||
      config.expected_candidate_sha.empty() || config.pair_count == 0 ||
      config.max_turns < 1 ||
      config.max_turns > 320 || !valid_sha(config.expected_bank_sha) ||
      !valid_sha(config.expected_candidate_sha) || config.candidate_c < 0.0 ||
      config.candidate_lambda < 0.0 || config.rank4_nodes == 0) {
    throw std::invalid_argument("invalid or incomplete gate configuration");
  }
  (void)candidate_config(config);
  return config;
}

void self_test() {
  const std::string transcript = "4/7/5/2/23/1/7/61/2/7";
  const std::string bank_text =
      "# " + std::string(kBankSchema) + "\nopening_id\ttranscript\n" +
      "smoke-a\t" + transcript + "\nsmoke-b\t" + transcript + "/0\n";
  const std::vector<Opening> openings = parse_bank_text(bank_text);
  if (openings.size() != 2 || openings[0].physical_plies != 12 ||
      openings[1].physical_plies != 13) {
    throw std::runtime_error("rank4 gate bank parser self-test failed");
  }
  MatchState state = openings[0].state;
  const cv::Action emergency = cv::emergency_complete_action(state.compact, 17);
  apply_both(state, emergency.text());
  append_transcript(state.transcript, emergency.text());
  verify_lockstep(state.core, state.compact);
  Config broken_candidate;
  broken_candidate.candidate_actions = 0;
  MatchState fallback_state = openings[0].state;
  const Invocation fallback = invoke_candidate(
      fallback_state, true, broken_candidate);
  if (!fallback.failure.empty() || fallback.action.empty() ||
      !fallback.deadline_reached || fallback.milliseconds < 0.0) {
    throw std::runtime_error("rank4 gate emergency fallback self-test failed");
  }
  verify_lockstep(fallback_state.core, fallback_state.compact);
  bool rejected = false;
  try {
    (void)parse_bank_text("opening_id\ttranscript\nshort\t0/1\n");
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  if (!rejected || kRank4Sha256.size() != 64) {
    throw std::runtime_error("rank4 gate rejection self-test failed");
  }
  std::cout << "compact_value_bfm Rank-4 gate self-test passed\n";
}

}  // namespace

#ifndef COMPACT_VALUE_BFM_RANK4_GATE_NO_MAIN
int main(int argc, char **argv) {
  try {
    const Config config = arguments(argc, argv);
    if (config.self_test) {
      self_test();
      return 0;
    }
    std::uint64_t candidate_bytes = 0;
    std::uint64_t rank4_bytes = 0;
    const std::string candidate_sha =
        file_sha(config.candidate_source, &candidate_bytes);
    const std::string rank4_sha = file_sha(config.rank4_source, &rank4_bytes);
    if (rank4_sha != kRank4Sha256) {
      throw std::invalid_argument("maintained Rank-4 source SHA-256 mismatch");
    }
    if (!config.expected_candidate_sha.empty() &&
        candidate_sha != config.expected_candidate_sha) {
      throw std::invalid_argument("candidate source SHA-256 mismatch");
    }
    const Bank bank = load_bank(config);
    std::vector<GameResult> games;
    games.reserve(config.pair_count * 2U);
    for (std::size_t pair = 0; pair < config.pair_count; ++pair) {
      const std::size_t bank_index = config.pair_offset + pair;
      for (int candidate_player = 0; candidate_player < 2; ++candidate_player) {
        games.push_back(play(bank.openings[bank_index], bank_index,
                             candidate_player, config));
      }
    }
    const GateSummary summary = summarize(games, config);
    const std::string json = render(config, bank, candidate_sha,
                                    candidate_bytes, rank4_bytes, games, summary);
    if (!config.output.empty()) {
      std::ofstream output(config.output, std::ios::binary | std::ios::trunc);
      if (!output) throw std::runtime_error("cannot open gate output");
      output << json;
      output.flush();
      if (!output) throw std::runtime_error("cannot write gate output");
    }
    std::cout << json;
    return summary.passed ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "compact_value_bfm Rank-4 gate failure: " << error.what()
              << '\n';
    return 1;
  }
}
#endif
