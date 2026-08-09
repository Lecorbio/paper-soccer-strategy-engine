#define PAPER_SOCCER_NEURAL_PUCT_NO_MAIN
#include "bot.cpp"
#include "visit_targets.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace neural = papersoccer::neural_puct;

namespace {

using TurnTargets = std::vector<neural::PrimitiveVisitTarget>;

struct Opening {
  ps::GameState state;
  std::vector<std::string> turns;
  std::vector<std::optional<TurnTargets>> policy_targets;
};

struct SearchTotals {
  std::uint64_t searches{};
  std::uint64_t simulations{};
  std::uint64_t neural_evaluations{};
  std::uint64_t minimum_simulations{std::numeric_limits<std::uint64_t>::max()};
  std::uint64_t maximum_simulations{};

  void add(const neural::SearchStats &stats) {
    ++searches;
    simulations += stats.simulations;
    neural_evaluations += stats.neural_evaluations;
    minimum_simulations = std::min(minimum_simulations, stats.simulations);
    maximum_simulations = std::max(maximum_simulations, stats.simulations);
  }
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

std::uint64_t next_random(std::uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  std::uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::string random_complete_turn(ps::GameState &state,
                                 std::uint64_t &random_state) {
  if (ps::is_terminal(state)) {
    throw std::invalid_argument("cannot extend a terminal opening");
  }
  const ps::Player mover = state.to_move;
  std::string action;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> moves = ps::legal_moves(state);
    if (moves.empty()) {
      throw std::logic_error("non-terminal state has no legal move");
    }
    const ps::Move move = moves[next_random(random_state) % moves.size()];
    action.push_back(neural::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  return action;
}

Opening random_opening(std::uint64_t seed, int requested_turns) {
  for (std::uint64_t attempt = 0; attempt < 10'000; ++attempt) {
    std::uint64_t random_state =
        seed ^ ((attempt + 1U) * 0xd1342543de82ef95ULL);
    Opening opening{ps::make_initial_state(codingame_rules()), {}, {}};
    bool valid = true;
    for (int turn = 0; turn < requested_turns; ++turn) {
      opening.turns.push_back(
          random_complete_turn(opening.state, random_state));
      opening.policy_targets.push_back(std::nullopt);
      if (ps::is_terminal(opening.state)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      return opening;
    }
  }
  throw std::runtime_error("could not construct a non-terminal opening");
}

std::string choose_neural_turn(ps::GameState &state,
                               std::uint64_t simulation_budget,
                               std::size_t maximum_nodes,
                               SearchTotals &totals,
                               TurnTargets &targets) {
  neural::SearchConfig config;
  config.max_nodes = maximum_nodes;
  config.max_simulations = simulation_budget;
  neural::NeuralPuctSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  totals.add(search.stats());
  targets = neural::NeuralPuctTrainingAccess::collect(search, moves);
  if (search.stats().deadline_reached) {
    throw std::logic_error("fixed-simulation search reached a deadline");
  }
  if (search.stats().node_cap_reached) {
    throw std::logic_error("self-play node cap truncated neural search");
  }
  if (moves.empty()) {
    throw std::logic_error("neural PUCT returned an empty action");
  }
  if (targets.size() != moves.size()) {
    throw std::logic_error("neural PUCT visit targets do not match its action");
  }

  const ps::Player mover = state.to_move;
  std::string action;
  action.reserve(moves.size());
  for (const ps::Move move : moves) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("neural PUCT returned an overlong action");
    }
    action.push_back(neural::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("neural PUCT omitted a mandatory rebound");
  }
  return action;
}

void write_game(std::ostream &output, int game_index, std::uint64_t seed,
                int opening_turns, std::uint64_t simulation_budget,
                std::size_t maximum_nodes, int winner,
                const std::vector<std::string> &turns,
                const std::vector<std::optional<TurnTargets>> &policy_targets,
                const SearchTotals &totals) {
  if (policy_targets.size() != turns.size()) {
    throw std::logic_error("policy targets do not align with game turns");
  }
  output << "{\"schema\":\"papersoccer.selfplay.v1\",\"generator\":"
            "\"neural-puct-selfplay-v1\",\"game\":"
         << game_index << ",\"seed\":" << seed
         << ",\"opening_turns\":" << opening_turns
         << ",\"opening_policy\":\"deterministic-random-unlabelled-v1\""
         << ",\"teacher_start_turn\":" << opening_turns
         << ",\"simulations\":" << simulation_budget
         << ",\"max_nodes\":" << maximum_nodes
         << ",\"teacher\":\"neural_puct\""
         << ",\"teacher_model_sha256\":\""
         << neural::model_data::kArtifactSha256 << '\"'
         << ",\"teacher_model_kind\":\"" << neural::model_data::kModelKind
         << "\",\"winner\":" << winner
         << ",\"searches\":" << totals.searches
         << ",\"actual_simulations\":" << totals.simulations
         << ",\"minimum_turn_simulations\":"
         << (totals.searches == 0 ? 0 : totals.minimum_simulations)
         << ",\"maximum_turn_simulations\":" << totals.maximum_simulations
         << ",\"neural_evaluations\":" << totals.neural_evaluations
         << ",\"policy_target_schema\":"
            "\"canonical-primitive-root-visits-v1\""
         << ",\"turns\":[";
  for (std::size_t index = 0; index < turns.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << '\"' << turns[index] << '\"';
  }
  output << "],\"policy_targets\":[" << std::setprecision(9);
  for (std::size_t turn = 0; turn < policy_targets.size(); ++turn) {
    if (turn != 0) output << ',';
    if (!policy_targets[turn].has_value()) {
      output << "null";
      continue;
    }
    const TurnTargets &targets = *policy_targets[turn];
    if (targets.size() != turns[turn].size()) {
      throw std::logic_error("primitive targets do not align with an action");
    }
    output << '[';
    for (std::size_t edge = 0; edge < targets.size(); ++edge) {
      if (edge != 0) output << ',';
      const neural::PrimitiveVisitTarget &target = targets[edge];
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
    output << ']';
  }
  output << "]}\n";
}

int parse_positive(const char *text, std::string_view name) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

std::size_t selfplay_node_cap(std::uint64_t simulations) {
  constexpr std::size_t kMinimumNodes = 100'000;
  if (simulations > std::numeric_limits<std::size_t>::max() - 1'024U) {
    throw std::invalid_argument("simulation budget is too large");
  }
  return std::max(kMinimumNodes,
                  static_cast<std::size_t>(simulations) + 1'024U);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 6) {
      std::cerr << "usage: " << argv[0]
                << " OUTPUT.jsonl GAMES [SIMULATIONS] [MAX_TURNS] [SEED]\n";
      return 2;
    }
    const int game_count = parse_positive(argv[2], "games");
    const std::uint64_t simulation_budget =
        argc >= 4
            ? static_cast<std::uint64_t>(
                  parse_positive(argv[3], "simulation budget"))
            : 2'000U;
    const int maximum_turns =
        argc >= 5 ? parse_positive(argv[4], "maximum turns") : 400;
    const std::uint64_t root_seed =
        argc >= 6 ? static_cast<std::uint64_t>(std::stoull(argv[5]))
                  : 0x4e50'5543'5420'2608ULL;
    const std::size_t maximum_nodes = selfplay_node_cap(simulation_budget);

    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("could not open self-play output");
    }

    constexpr std::array<int, 8> kOpeningDepths{{0, 2, 4, 6, 8, 12, 16, 20}};
    int completed = 0;
    int attempts = 0;
    while (completed < game_count && attempts < game_count * 50) {
      const std::uint64_t game_seed =
          root_seed + static_cast<std::uint64_t>(attempts) *
                          0x9e3779b97f4a7c15ULL;
      std::uint64_t selector = game_seed ^ 0x6a09e667f3bcc909ULL;
      const int opening_turns =
          completed == 0
              ? 0
              : kOpeningDepths[next_random(selector) % kOpeningDepths.size()];
      Opening game = random_opening(game_seed, opening_turns);
      SearchTotals totals;
      int played_turns = opening_turns;
      while (!ps::is_terminal(game.state) && played_turns < maximum_turns) {
        TurnTargets targets;
        game.turns.push_back(choose_neural_turn(
            game.state, simulation_budget, maximum_nodes, totals, targets));
        game.policy_targets.emplace_back(std::move(targets));
        ++played_turns;
      }
      const std::optional<ps::Player> winning = ps::winner(game.state);
      if (winning.has_value()) {
        write_game(output, completed, game_seed, opening_turns,
                   simulation_budget, maximum_nodes, player_id(*winning),
                   game.turns, game.policy_targets, totals);
        ++completed;
        std::cerr << "completed " << completed << '/' << game_count
                  << " games\n";
      }
      ++attempts;
    }
    if (completed != game_count) {
      throw std::runtime_error("too many unfinished self-play games");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "neural self-play generator: " << error.what() << '\n';
    return 1;
  }
}
