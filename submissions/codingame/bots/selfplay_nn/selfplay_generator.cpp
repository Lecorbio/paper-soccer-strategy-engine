#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 selfplay_teacher
#include "../rank_5/bot.cpp"
#undef turn_action_v2

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace teacher = papersoccer::selfplay_teacher;

namespace {

struct Opening {
  ps::GameState state;
  std::vector<std::string> turns;
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
    throw std::invalid_argument("cannot play from a terminal state");
  }
  const ps::Player mover = state.to_move;
  std::string action;
  while (!ps::is_terminal(state) && state.to_move == mover) {
    const std::vector<ps::Move> moves = ps::legal_moves(state);
    if (moves.empty()) {
      throw std::logic_error("non-terminal state has no legal move");
    }
    const ps::Move move = moves[next_random(random_state) % moves.size()];
    action.push_back(teacher::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  return action;
}

Opening random_opening(std::uint64_t seed, int requested_turns) {
  for (std::uint64_t attempt = 0; attempt < 10'000; ++attempt) {
    std::uint64_t random_state =
        seed ^ ((attempt + 1U) * 0xd1342543de82ef95ULL);
    Opening opening{ps::make_initial_state(codingame_rules()), {}};
    bool valid = true;
    for (int turn = 0; turn < requested_turns; ++turn) {
      const std::string action =
          random_complete_turn(opening.state, random_state);
      opening.turns.push_back(action);
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

std::string choose_teacher_turn(ps::GameState &state,
                                std::uint64_t node_budget) {
  teacher::SearchConfig config;
  config.max_nodes = node_budget;
  config.transposition_entries = 32'768;
  config.evaluation_entries = 16'384;
  config.replay_value_blend_percent = 15;
  teacher::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  if (moves.empty()) {
    throw std::logic_error("teacher returned an empty action");
  }

  const ps::Player mover = state.to_move;
  std::string action;
  action.reserve(moves.size());
  for (const ps::Move move : moves) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("teacher returned an overlong action");
    }
    action.push_back(teacher::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("teacher omitted a mandatory rebound");
  }
  return action;
}

void write_game(std::ostream &output, int game_index, std::uint64_t seed,
                int opening_turns, std::uint64_t node_budget, int winner,
                const std::vector<std::string> &turns) {
  output << "{\"schema\":\"papersoccer.selfplay.v1\",\"game\":"
         << game_index << ",\"seed\":" << seed
         << ",\"opening_turns\":" << opening_turns
         << ",\"teacher_start_turn\":" << opening_turns
         << ",\"node_budget\":" << node_budget
         << ",\"teacher\":\"rank_5\",\"winner\":" << winner
         << ",\"turns\":[";
  for (std::size_t index = 0; index < turns.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << '"' << turns[index] << '"';
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

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 6) {
      std::cerr << "usage: " << argv[0]
                << " OUTPUT.jsonl GAMES [NODE_BUDGET] [MAX_TURNS] [SEED]\n";
      return 2;
    }
    const int game_count = parse_positive(argv[2], "games");
    const std::uint64_t base_budget =
        argc >= 4 ? static_cast<std::uint64_t>(parse_positive(argv[3], "budget"))
                  : 4'000U;
    const int maximum_turns =
        argc >= 5 ? parse_positive(argv[4], "maximum turns") : 180;
    const std::uint64_t root_seed =
        argc >= 6 ? static_cast<std::uint64_t>(std::stoull(argv[5]))
                  : 0x5e1f'9a7d'2026'0722ULL;

    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("could not open self-play output");
    }

    constexpr std::array<int, 6> kOpeningDepths{{4, 6, 8, 12, 16, 20}};
    int completed = 0;
    int attempts = 0;
    while (completed < game_count && attempts < game_count * 30) {
      const std::uint64_t game_seed =
          root_seed + static_cast<std::uint64_t>(attempts) *
                          0x9e3779b97f4a7c15ULL;
      std::uint64_t selector = game_seed ^ 0x6a09e667f3bcc909ULL;
      const int opening_turns =
          completed == 0
              ? 0
              : kOpeningDepths[next_random(selector) % kOpeningDepths.size()];
      constexpr std::array<std::uint64_t, 3> kBudgetMultipliers{{1, 2, 4}};
      const std::uint64_t budget_multiplier =
          kBudgetMultipliers[next_random(selector) %
                             kBudgetMultipliers.size()];
      const std::uint64_t node_budget =
          std::max<std::uint64_t>(500U,
                                  base_budget * budget_multiplier / 2U);
      Opening game = random_opening(game_seed, opening_turns);
      int played_turns = opening_turns;
      while (!ps::is_terminal(game.state) && played_turns < maximum_turns) {
        game.turns.push_back(choose_teacher_turn(game.state, node_budget));
        ++played_turns;
      }
      const std::optional<ps::Player> winning = ps::winner(game.state);
      if (winning.has_value()) {
        write_game(output, completed, game_seed, opening_turns, node_budget,
                   player_id(*winning), game.turns);
        ++completed;
        if (completed % 16 == 0 || completed == game_count) {
          std::cerr << "completed " << completed << '/' << game_count
                    << " games\n";
        }
      }
      ++attempts;
    }
    if (completed != game_count) {
      throw std::runtime_error("too many unfinished self-play games");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "self-play generator: " << error.what() << '\n';
    return 1;
  }
}
