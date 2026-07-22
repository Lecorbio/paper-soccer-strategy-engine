#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 teacher_engine
#include "bot.cpp"
#undef turn_action_v2

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace teacher = papersoccer::teacher_engine;

namespace {

struct Opening {
  ps::GameState state;
  std::vector<std::string> turns;
};

struct Sample {
  int player_id{};
  std::string transcript;
  std::string action;
  std::uint64_t node_budget{};
  int anchor_score{};
  int teacher_score{};
  std::uint32_t completed_depth{};
  std::uint32_t attempted_depth{};
  std::uint64_t nodes{};
  bool budget_exhausted{};
  std::array<float, teacher::kTeacherResidualInputs> features{};
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

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) {
    transcript.push_back('/');
  }
  transcript.append(action);
}

std::string random_complete_turn(ps::GameState &state,
                                 std::uint64_t &random_state) {
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
      opening.turns.push_back(
          random_complete_turn(opening.state, random_state));
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

Sample choose_teacher_turn(ps::GameState &state, std::string transcript,
                           std::uint64_t node_budget) {
  teacher::SearchConfig config;
  config.max_nodes = node_budget;
  config.transposition_entries = 65'536;
  config.evaluation_entries = 32'768;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 0;
  teacher::CompleteTurnSearch search(state, config);
  const ps::Player mover = state.to_move;
  const teacher::EvaluationSnapshot snapshot = search.evaluation_snapshot();
  const std::vector<ps::Move> moves = search.run();
  if (moves.empty()) {
    throw std::logic_error("teacher returned an empty action");
  }

  Sample sample;
  sample.player_id = player_id(mover);
  sample.transcript = std::move(transcript);
  sample.node_budget = node_budget;
  sample.anchor_score = snapshot.anchor_score;
  sample.features = snapshot.features;
  const teacher::SearchStats stats = search.stats();
  sample.teacher_score = stats.root_score;
  sample.completed_depth = stats.completed_turn_depth;
  sample.attempted_depth = stats.attempted_turn_depth;
  sample.nodes = stats.nodes;
  sample.budget_exhausted = stats.budget_exhausted;

  for (const ps::Move move : moves) {
    if (ps::is_terminal(state) || state.to_move != mover) {
      throw std::logic_error("teacher returned an overlong action");
    }
    sample.action.push_back(teacher::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  if (!ps::is_terminal(state) && state.to_move == mover) {
    throw std::logic_error("teacher omitted a mandatory rebound");
  }
  return sample;
}

void write_game(std::ostream &output, int game, std::uint64_t seed,
                int opening_turns, int winner,
                const std::vector<Sample> &samples) {
  output << "{\"schema\":\"papersoccer.teacher-residual-samples.v1\""
         << ",\"game\":" << game << ",\"seed\":" << seed
         << ",\"opening_turns\":" << opening_turns
         << ",\"winner\":" << winner << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    const Sample &sample = samples[index];
    output << "{\"player_id\":" << sample.player_id
           << ",\"transcript\":\"" << sample.transcript
           << "\",\"action\":\"" << sample.action
           << "\",\"node_budget\":" << sample.node_budget
           << ",\"anchor_score\":" << sample.anchor_score
           << ",\"teacher_score\":" << sample.teacher_score
           << ",\"completed_depth\":" << sample.completed_depth
           << ",\"attempted_depth\":" << sample.attempted_depth
           << ",\"nodes\":" << sample.nodes
           << ",\"budget_exhausted\":"
           << (sample.budget_exhausted ? "true" : "false")
           << ",\"features\":[";
    for (std::size_t feature = 0; feature < sample.features.size();
         ++feature) {
      if (feature != 0) {
        output << ',';
      }
      output << std::setprecision(9) << sample.features[feature];
    }
    output << "]}";
  }
  output << "]}\n";
}

int parse_positive(const char *text, std::string_view label) {
  const int value = std::stoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(label) + " must be positive");
  }
  return value;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3 || argc > 6) {
      std::cerr << "usage: " << argv[0]
                << " OUTPUT.jsonl GAMES [BASE_BUDGET] [MAX_TURNS] [SEED]\n";
      return 2;
    }
    const int game_count = parse_positive(argv[2], "games");
    const std::uint64_t base_budget =
        argc >= 4 ? static_cast<std::uint64_t>(
                        parse_positive(argv[3], "base budget"))
                  : 16'000U;
    const int maximum_turns =
        argc >= 5 ? parse_positive(argv[4], "maximum turns") : 160;
    const std::uint64_t root_seed =
        argc >= 6 ? static_cast<std::uint64_t>(std::stoull(argv[5]))
                  : 0x76a4'29d1'2026'0722ULL;

    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    if (!output) {
      throw std::runtime_error("could not open teacher sample output");
    }

    constexpr std::array<int, 7> kOpeningDepths{{2, 4, 6, 8, 12, 16, 20}};
    constexpr std::array<std::uint64_t, 3> kBudgetMultipliers{{1, 2, 4}};
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
      Opening game = random_opening(game_seed, opening_turns);
      std::string transcript;
      for (const std::string &turn : game.turns) {
        append_turn(transcript, turn);
      }
      std::vector<Sample> samples;
      int played_turns = opening_turns;
      while (!ps::is_terminal(game.state) && played_turns < maximum_turns) {
        const std::uint64_t multiplier =
            kBudgetMultipliers[next_random(selector) %
                               kBudgetMultipliers.size()];
        const std::uint64_t budget = base_budget * multiplier / 2U;
        Sample sample = choose_teacher_turn(game.state, transcript, budget);
        append_turn(transcript, sample.action);
        samples.push_back(std::move(sample));
        ++played_turns;
      }
      const std::optional<ps::Player> winning = ps::winner(game.state);
      if (winning.has_value() && !samples.empty()) {
        write_game(output, completed, game_seed, opening_turns,
                   player_id(*winning), samples);
        ++completed;
        if (completed % 16 == 0 || completed == game_count) {
          std::cerr << "completed " << completed << '/' << game_count
                    << " games\n";
        }
      }
      ++attempts;
    }
    if (completed != game_count) {
      throw std::runtime_error("too many unfinished teacher games");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "teacher sample generator: " << error.what() << '\n';
    return 1;
  }
}
