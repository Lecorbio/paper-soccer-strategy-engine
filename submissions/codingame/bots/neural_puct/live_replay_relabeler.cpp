#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "bot.cpp"
#include "visit_targets.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace neural = papersoccer::neural_puct;

namespace {

struct Turn {
  int player{};
  std::string action;
};

struct Target {
  neural::PrimitiveVisitTarget policy;
  float value{};
  float played_probability{};
  float normalized_entropy{};
  float priority{};
  bool disagreement{};
  bool precedes_tactical_punishment{};
  std::uint64_t simulations{};
  std::uint64_t neural_evaluations{};
};

struct GameInput {
  std::uint64_t game_id{};
  std::string record_sha256;
  int winner{};
  std::uint64_t own_agent_id{};
  int own_player{};
  std::vector<Turn> turns;
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

std::vector<std::string> split(std::string_view text, char separator) {
  std::vector<std::string> result;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find(separator, start);
    result.emplace_back(text.substr(
        start, end == std::string_view::npos ? text.size() - start
                                             : end - start));
    if (end == std::string_view::npos) break;
    start = end + 1;
  }
  return result;
}

int parse_player(std::string_view text) {
  if (text == "0") return 0;
  if (text == "1") return 1;
  throw std::invalid_argument("player id must be zero or one");
}

GameInput parse_game(std::string_view line) {
  const std::vector<std::string> fields = split(line, '\t');
  if (fields.size() != 6) {
    throw std::invalid_argument("relabel input row must have six fields");
  }
  GameInput result;
  result.game_id = std::stoull(fields[0]);
  result.record_sha256 = fields[1];
  if (result.record_sha256.size() != 64 ||
      !std::all_of(result.record_sha256.begin(), result.record_sha256.end(),
                   [](char value) {
                     return (value >= '0' && value <= '9') ||
                            (value >= 'a' && value <= 'f');
                   })) {
    throw std::invalid_argument("source record hash is invalid");
  }
  result.winner = parse_player(fields[2]);
  result.own_agent_id = std::stoull(fields[3]);
  result.own_player = parse_player(fields[4]);
  for (const std::string &encoded : split(fields[5], '/')) {
    const std::size_t colon = encoded.find(':');
    if (colon != 1 || encoded.size() < 3) {
      throw std::invalid_argument("turn encoding is invalid");
    }
    Turn turn{parse_player(std::string_view(encoded).substr(0, 1)),
              encoded.substr(2)};
    if (!std::all_of(turn.action.begin(), turn.action.end(), [](char value) {
          return value >= '0' && value <= '7';
        })) {
      throw std::invalid_argument("turn contains a non-direction character");
    }
    result.turns.push_back(std::move(turn));
  }
  if (result.turns.empty()) {
    throw std::invalid_argument("game has no turns");
  }
  return result;
}

std::size_t node_cap(std::uint64_t simulations, std::size_t requested) {
  if (requested != 0) return requested;
  if (simulations > std::numeric_limits<std::size_t>::max() - 1'024U) {
    throw std::invalid_argument("simulation budget is too large");
  }
  return std::max<std::size_t>(100'000, simulations + 1'024U);
}

Target relabel(const ps::GameState &state, char played,
               std::uint64_t simulations, std::size_t maximum_nodes,
               bool punishment) {
  neural::SearchConfig config;
  config.max_simulations = simulations;
  config.max_nodes = maximum_nodes;
  neural::NeuralPuctSearch search(state, config);
  (void)search.run();
  if (search.stats().deadline_reached || search.stats().node_cap_reached) {
    throw std::runtime_error("deep relabel search was truncated");
  }
  Target result;
  result.policy = neural::NeuralPuctTrainingAccess::root(search);
  result.value = search.stats().root_value;
  result.simulations = search.stats().simulations;
  result.neural_evaluations = search.stats().neural_evaluations;
  const ps::Move played_move = neural::decode_direction(state.ball, played);
  const std::size_t played_direction = neural::canonical_direction(
      state.ball, played_move.to, state.to_move);
  result.played_probability = result.policy.probabilities[played_direction];
  const auto maximum = std::max_element(result.policy.probabilities.begin(),
                                        result.policy.probabilities.end());
  result.disagreement =
      static_cast<std::size_t>(maximum - result.policy.probabilities.begin()) !=
      played_direction;
  const std::size_t legal_count = ps::legal_moves(state).size();
  float entropy = 0.0F;
  for (const float probability : result.policy.probabilities) {
    if (probability > 0.0F) entropy -= probability * std::log(probability);
  }
  result.normalized_entropy =
      legal_count <= 1 ? 0.0F
                       : entropy / std::log(static_cast<float>(legal_count));
  result.precedes_tactical_punishment = punishment;
  result.priority = 1.0F + (result.disagreement ? 2.0F : 0.0F) +
                    result.normalized_entropy +
                    (1.0F - result.played_probability) +
                    (punishment ? 1.0F : 0.0F);
  return result;
}

void print_policy(std::ostream &output,
                  const neural::PrimitiveVisitTarget &target) {
  output << "{\"probabilities\":[";
  for (std::size_t direction = 0; direction < target.probabilities.size();
       ++direction) {
    if (direction != 0) output << ',';
    output << target.probabilities[direction];
  }
  output << "],\"total_visits\":" << target.total_visits
         << ",\"fallback\":" << (target.fallback ? "true" : "false")
         << '}';
}

void write_game(std::ostream &output, const GameInput &game,
                const std::vector<std::optional<std::vector<Target>>> &targets,
                std::uint64_t requested_simulations,
                std::size_t maximum_nodes) {
  std::uint64_t searches = 0;
  std::uint64_t actual_simulations = 0;
  std::uint64_t neural_evaluations = 0;
  for (const auto &turn : targets) {
    if (!turn.has_value()) continue;
    for (const Target &target : *turn) {
      ++searches;
      actual_simulations += target.simulations;
      neural_evaluations += target.neural_evaluations;
    }
  }
  output << std::setprecision(9)
         << "{\"schema\":\"papersoccer.live-replay-relabel.v1\""
         << ",\"generator\":\"neural-puct-live-relabel-v1\""
         << ",\"game_id\":" << game.game_id
         << ",\"source_record_sha256\":\"" << game.record_sha256 << '"'
         << ",\"own_agent_id\":" << game.own_agent_id
         << ",\"own_player_id\":" << game.own_player
         << ",\"winner\":" << game.winner
         << ",\"teacher\":\"neural_puct\""
         << ",\"teacher_model_sha256\":\""
         << neural::model_data::kArtifactSha256 << '"'
         << ",\"teacher_model_kind\":\"" << neural::model_data::kModelKind
         << "\",\"requested_simulations\":" << requested_simulations
         << ",\"max_nodes\":" << maximum_nodes
         << ",\"searches\":" << searches
         << ",\"actual_simulations\":" << actual_simulations
         << ",\"neural_evaluations\":" << neural_evaluations
         << ",\"policy_target_schema\":"
            "\"canonical-primitive-root-visits-v1\""
         << ",\"value_target_schema\":"
            "\"neural-puct-root-mover-expectation-v1\""
         << ",\"turn_players\":[";
  for (std::size_t turn = 0; turn < game.turns.size(); ++turn) {
    if (turn != 0) output << ',';
    output << game.turns[turn].player;
  }
  output << "],\"turns\":[";
  for (std::size_t turn = 0; turn < game.turns.size(); ++turn) {
    if (turn != 0) output << ',';
    output << '"' << game.turns[turn].action << '"';
  }
  output << "],\"policy_targets\":[";
  for (std::size_t turn = 0; turn < targets.size(); ++turn) {
    if (turn != 0) output << ',';
    if (!targets[turn].has_value()) {
      output << "null";
      continue;
    }
    output << '[';
    for (std::size_t edge = 0; edge < targets[turn]->size(); ++edge) {
      if (edge != 0) output << ',';
      print_policy(output, (*targets[turn])[edge].policy);
    }
    output << ']';
  }
  output << "],\"value_targets\":[";
  for (std::size_t turn = 0; turn < targets.size(); ++turn) {
    if (turn != 0) output << ',';
    if (!targets[turn].has_value()) {
      output << "null";
      continue;
    }
    output << '[';
    for (std::size_t edge = 0; edge < targets[turn]->size(); ++edge) {
      if (edge != 0) output << ',';
      output << (*targets[turn])[edge].value;
    }
    output << ']';
  }
  output << "],\"priorities\":[";
  for (std::size_t turn = 0; turn < targets.size(); ++turn) {
    if (turn != 0) output << ',';
    if (!targets[turn].has_value()) {
      output << "null";
      continue;
    }
    output << '[';
    for (std::size_t edge = 0; edge < targets[turn]->size(); ++edge) {
      if (edge != 0) output << ',';
      const Target &target = (*targets[turn])[edge];
      output << "{\"played_probability\":" << target.played_probability
             << ",\"normalized_entropy\":" << target.normalized_entropy
             << ",\"disagreement\":"
             << (target.disagreement ? "true" : "false")
             << ",\"precedes_tactical_punishment\":"
             << (target.precedes_tactical_punishment ? "true" : "false")
             << ",\"weight\":" << target.priority << '}';
    }
    output << ']';
  }
  output << "]}\n";
}

void relabel_game(std::ostream &output, const GameInput &game,
                  std::uint64_t simulations, std::size_t maximum_nodes) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  std::vector<std::optional<std::vector<Target>>> targets;
  targets.reserve(game.turns.size());
  for (std::size_t turn_index = 0; turn_index < game.turns.size();
       ++turn_index) {
    const Turn &turn = game.turns[turn_index];
    if (neural::player_for_id(turn.player) != state.to_move ||
        ps::is_terminal(state)) {
      throw std::runtime_error("recorded turn disagrees with replay state");
    }
    const bool punishment =
        game.winner != game.own_player &&
        turn_index + 1 < game.turns.size() &&
        game.turns[turn_index + 1].player != game.own_player &&
        (turn_index + 2 == game.turns.size() ||
         game.turns[turn_index + 1].action.size() >= 4);
    std::optional<std::vector<Target>> turn_targets;
    if (turn.player == game.own_player) turn_targets.emplace();
    const ps::Player mover = state.to_move;
    for (std::size_t edge = 0; edge < turn.action.size(); ++edge) {
      const char direction = turn.action[edge];
      if (turn_targets.has_value()) {
        turn_targets->push_back(relabel(state, direction, simulations,
                                        maximum_nodes, punishment));
      }
      const ps::Move move = neural::decode_direction(state.ball, direction);
      state = ps::apply_move(state, move);
      if (edge + 1 < turn.action.size() &&
          (ps::is_terminal(state) || state.to_move != mover)) {
        throw std::runtime_error("recorded turn is overlong");
      }
    }
    if (!ps::is_terminal(state) && state.to_move == mover) {
      throw std::runtime_error("recorded turn omits a mandatory rebound");
    }
    targets.push_back(std::move(turn_targets));
  }
  const std::optional<ps::Player> winner = ps::winner(state);
  if (!winner.has_value() || neural::player_for_id(game.winner) != *winner) {
    throw std::runtime_error("recorded winner disagrees with replay");
  }
  write_game(output, game, targets, simulations, maximum_nodes);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 4 || argc > 5) {
      std::cerr << "usage: " << argv[0]
                << " INPUT.tsv OUTPUT.jsonl SIMULATIONS [MAX_NODES]\n";
      return 2;
    }
    const std::uint64_t simulations = std::stoull(argv[3]);
    if (simulations == 0) {
      throw std::invalid_argument("simulations must be positive");
    }
    const std::size_t maximum_nodes =
        node_cap(simulations, argc == 5 ? std::stoull(argv[4]) : 0U);
    if (maximum_nodes == 0 || maximum_nodes > neural::kMaximumNodes) {
      throw std::invalid_argument("max nodes exceeds the production hard cap");
    }
    std::ifstream input(argv[1], std::ios::binary);
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!input || !output) {
      throw std::runtime_error("could not open relabel input or output");
    }
    std::string line;
    if (!std::getline(input, line)) {
      throw std::invalid_argument("relabel input is empty");
    }
    const std::vector<std::string> header = split(line, '\t');
    if (header.size() != 4 ||
        header[0] != "papersoccer.live-replay-relabel-input.v1") {
      throw std::invalid_argument("relabel input header is invalid");
    }
    std::size_t games = 0;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      relabel_game(output, parse_game(line), simulations, maximum_nodes);
      ++games;
    }
    if (games != std::stoull(header[3])) {
      throw std::runtime_error("relabel input game count disagrees with header");
    }
    std::cout << "Relabelled " << games << " games at " << simulations
              << " simulations per owned primitive.\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "live replay relabel failed: " << error.what() << '\n';
    return 1;
  }
}
