// State-capable local-opponent adapter. Deployment stdin/stdout is unchanged.
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN
#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#define turn_action_v2 campaign_opponent
#define jacek_native_bfm campaign_opponent
#define neural_puct campaign_opponent
#ifndef CAMPAIGN_OPPONENT_SOURCE
#error "CAMPAIGN_OPPONENT_SOURCE must identify the frozen opponent bot.cpp"
#endif
#include CAMPAIGN_OPPONENT_SOURCE
#undef turn_action_v2
#undef jacek_native_bfm
#undef neural_puct
#define COMPACT_VALUE_BFM_NO_MAIN
#ifndef CAMPAIGN_CANDIDATE_SOURCE
#error "CAMPAIGN_CANDIDATE_SOURCE must identify the exact candidate source"
#endif
#include CAMPAIGN_CANDIDATE_SOURCE
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ps = papersoccer;
namespace opponent = papersoccer::campaign_opponent;
namespace cv = compact_value_bfm;
using Clock = std::chrono::steady_clock;
namespace {
struct AdapterMismatch : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct MatchState { ps::GameState core; cv::State compact; std::string transcript; };
ps::RulesConfig codingame_rules() {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  return rules;
}

int player_id(ps::Player player) noexcept {
  return player == ps::Player::One ? 0 : 1;
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
    throw AdapterMismatch("core/compact dynamic-state mismatch");
  }
  const std::optional<ps::Player> winner = ps::winner(core);
  if ((winner.has_value() ? player_id(*winner) : -1) != compact.winner) {
    throw AdapterMismatch("core/compact winner mismatch");
  }
  std::size_t compact_edges = 0;
  for (int edge = 0; edge < cv::kEdgeCount; ++edge) {
    if (!cv::edge_used(compact, edge)) continue;
    ++compact_edges;
    const auto [first, second] = compact_edge_points(edge);
    if (!core.used_segments.contains(ps::Segment{first, second})) {
      throw AdapterMismatch("core/compact used-edge mismatch");
    }
  }
  if (compact_edges != core.used_segments.size()) {
    throw AdapterMismatch("core/compact used-edge count mismatch");
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
      throw AdapterMismatch("core/compact visited-vertex mismatch");
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
  opponent::apply_encoded_turn(next_core, encoded);
  if (!cv::apply_action(next_compact, compact_action(encoded))) {
    throw AdapterMismatch("compact engine rejected core-legal action");
  }
  verify_lockstep(next_core, next_compact);
  state.core = std::move(next_core);
  state.compact = next_compact;
}


MatchState root_state(const std::string &transcript) {
  MatchState state{ps::make_initial_state(codingame_rules()), cv::initial_state(), {}};
  if (transcript.empty() || transcript == "-")
    throw std::invalid_argument("empty-board games must use the process referee");
  std::istringstream stream(transcript);
  std::string action;
  while (std::getline(stream, action, '/')) apply_both(state, action);
  if (ps::is_terminal(state.core)) throw std::invalid_argument("terminal root");
  verify_lockstep(state.core, state.compact);
  state.transcript=transcript;
  return state;
}
void play(const std::string &id, const MatchState &root, int candidate_player,
          unsigned first_ms, unsigned later_ms) {
  MatchState state=root;
  std::array<unsigned,2> decisions{};
  unsigned turns=0;
  double maximum_ms=0;
  std::uint64_t generated=0, evaluated=0, nodes=0;
  std::vector<double> latencies;
  std::string failure;
  while (!state.compact.terminal() && turns<320) {
    const bool candidate=state.compact.to_move==candidate_player;
    const unsigned actor=candidate ? 0 : 1;
    const bool first=decisions[actor]++==0;
    const bool production_clocks=first_ms==800 && later_ms==155;
    const unsigned budget=candidate || !production_clocks ? (first ? first_ms : later_ms)
        : (first ? opponent::kFirstSearchTimeMs : opponent::kLaterSearchTimeMs);
    const auto started=Clock::now();
    std::string action;
    try {
      if (candidate) {
        const auto emergency=cv::emergency_complete_action(state.compact);
        cv::SearchConfig config;
        const auto answer=cv::search(state.compact, started+std::chrono::milliseconds(budget),config,&emergency);
        action=answer.action.text();
        generated+=answer.stats.generated_children;
        evaluated+=answer.stats.evaluated_children;
        nodes+=answer.stats.tree_nodes;
      } else {
        auto copy=state.core;
#if defined(CAMPAIGN_OPPONENT_REPLAY_CORRECTION)
        if (!opponent::try_replay_correction(copy,1-candidate_player,state.transcript,action))
#endif
          action=opponent::choose_complete_turn(copy,budget);
      }
      const double ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
      if (candidate) { latencies.push_back(ms); maximum_ms=std::max(maximum_ms,ms); }
      if (ms>(first ? 1000.0 : 200.0)) {
        failure=candidate ? "candidate-timeout" : "opponent-timeout"; break;
      }
      apply_both(state,action);
      append_transcript(state.transcript,action);
      ++turns;
    } catch (const AdapterMismatch &) {
      failure="adapter-state-mismatch";
      break;
    } catch (const std::exception &) {
      failure=candidate ? "candidate-exception-or-illegal" : "opponent-exception-or-illegal";
      break;
    }
  }
  if (failure.empty() && !state.compact.terminal()) failure="unfinished";
  std::cout << "{\"schema\":\"papersoccer.compact-state-evaluation.v2\",\"root_id\":\"" << id
    << "\",\"candidate_player\":" << candidate_player << ",\"winner\":" << int(state.compact.winner)
    << ",\"failure\":\"" << failure << "\",\"turns\":" << turns
    << ",\"root_transcript\":\"" << root.transcript << "\""
    << ",\"root_edges\":" << root.compact.ply << ",\"candidate_max_ms\":" << maximum_ms
    << ",\"first_budget_ms\":" << first_ms << ",\"later_budget_ms\":" << later_ms
    << ",\"opponent_first_budget_ms\":" << (first_ms==800 && later_ms==155 ? opponent::kFirstSearchTimeMs : first_ms)
    << ",\"opponent_later_budget_ms\":" << (first_ms==800 && later_ms==155 ? opponent::kLaterSearchTimeMs : later_ms)
    << ",\"generated_successors\":" << generated << ",\"evaluated_successors\":" << evaluated
    << ",\"nodes\":" << nodes << ",\"candidate_latency_ms\":[";
  for (std::size_t i=0;i<latencies.size();++i) { if (i) std::cout << ','; std::cout << latencies[i]; }
  std::cout << "],\"trajectory\":\"" << state.transcript << "\"}\n" << std::flush;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc!=4) throw std::invalid_argument("usage: adapter ROOTS_TSV FIRST_MS LATER_MS");
    unsigned first=std::stoul(argv[2]), later=std::stoul(argv[3]);
    if (!first || !later || first>800 || later>155) throw std::invalid_argument("invalid internal clocks");
    std::ifstream input(argv[1]);
    if (!input) throw std::invalid_argument("root bank missing");
    std::string line;
    std::cout << std::setprecision(17);
    while (std::getline(input,line)) {
      if (line=="root_id\ttranscript") continue;
      auto tab=line.find('\t');
      if (tab==std::string::npos) throw std::invalid_argument("malformed root row");
      auto id=line.substr(0,tab);
      if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789:-_.")!=std::string::npos)
        throw std::invalid_argument("invalid root id");
      const auto root=root_state(line.substr(tab+1));
      for (int color=0;color<2;++color) play(id,root,color,first,later);
    }
    return 0;
  } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 1; }
}
