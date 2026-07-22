#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank5_engine
#include "../rank_5/bot.cpp"
#undef turn_action_v2

#define PAPER_SOCCER_SELFPLAY_NN_NO_MAIN
#include "bot.cpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace rank5 = papersoccer::rank5_engine;
namespace candidate = papersoccer::selfplay_nn;

namespace {

int candidate_value_blend = 15;
int candidate_policy_weight = 0;
std::uint32_t game_time_ms = 0;
bool reference_replay_corrections = false;
bool candidate_replay_corrections = true;

struct Decision {
  std::string action;
  std::uint64_t nodes{};
  std::uint32_t completed_depth{};
  int root_score{};
  double milliseconds{};
  std::uint64_t neural_evaluations{};
  std::uint64_t policy_orderings{};
};

struct Opening {
  ps::GameState state;
  std::string transcript;
  std::string label;
};

struct GameResult {
  int winner{-1};
  int turns{};
  std::uint64_t candidate_nodes{};
  std::uint64_t reference_nodes{};
  std::uint64_t candidate_neural_evaluations{};
};

struct Summary {
  int candidate_wins{};
  int reference_wins{};
  int unfinished{};
  int candidate_as_zero_wins{};
  int candidate_as_one_wins{};
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

int player_id(ps::Player player) {
  return player == ps::Player::One ? 0 : 1;
}

void append_turn(std::string &transcript, std::string_view action) {
  if (!transcript.empty()) {
    transcript.push_back('/');
  }
  transcript.append(action);
}

void verify_complete_action(const ps::GameState &before,
                            const ps::GameState &after,
                            std::string_view action) {
  if (action.empty()) {
    throw std::logic_error("engine returned an empty action");
  }
  ps::GameState replayed = before;
  rank5::apply_encoded_turn(replayed, action);
  if (replayed.ball != after.ball || replayed.to_move != after.to_move ||
      replayed.status != after.status ||
      replayed.used_segments != after.used_segments ||
      replayed.visit_count != after.visit_count) {
    throw std::logic_error("engine action did not reproduce its state");
  }
}

Decision choose_rank5(ps::GameState &state, std::string_view transcript,
                      std::uint64_t node_budget,
                      bool replay_corrections = true,
                      std::uint32_t time_budget_ms = 0) {
  const ps::GameState before = state;
  Decision decision;
  if (!replay_corrections ||
      !rank5::try_replay_correction(state, player_id(state.to_move), transcript,
                                    decision.action)) {
    rank5::SearchConfig config;
    config.replay_value_blend_percent = 15;
    const auto started = rank5::SearchClock::now();
    if (time_budget_ms == 0) {
      config.max_nodes = node_budget;
    } else {
      config.absolute_deadline =
          started + std::chrono::milliseconds(time_budget_ms);
    }
    rank5::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
    decision.milliseconds =
        std::chrono::duration<double, std::milli>(
            rank5::SearchClock::now() - started)
            .count();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error("rank-5 search returned an overlong action");
      }
      decision.action.push_back(rank5::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    decision.nodes = search.stats().nodes;
    decision.completed_depth = search.stats().completed_turn_depth;
    decision.root_score = search.stats().root_score;
  }
  verify_complete_action(before, state, decision.action);
  return decision;
}

Decision choose_candidate(ps::GameState &state, std::string_view transcript,
                          std::uint64_t node_budget,
                          std::uint32_t time_budget_ms = 0) {
  const ps::GameState before = state;
  Decision decision;
  if (!candidate_replay_corrections ||
      !candidate::try_replay_correction(state, player_id(state.to_move),
                                        transcript, decision.action)) {
    candidate::SearchConfig config;
    config.neural_value_blend_percent = candidate_value_blend;
    config.neural_policy_order_weight = candidate_policy_weight;
    const auto started = candidate::SearchClock::now();
    if (time_budget_ms == 0) {
      config.max_nodes = node_budget;
    } else {
      config.absolute_deadline =
          started + std::chrono::milliseconds(time_budget_ms);
    }
    candidate::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
    decision.milliseconds =
        std::chrono::duration<double, std::milli>(
            candidate::SearchClock::now() - started)
            .count();
    const ps::Player mover = state.to_move;
    for (const ps::Move move : moves) {
      if (ps::is_terminal(state) || state.to_move != mover) {
        throw std::logic_error(
            "candidate search returned an overlong action");
      }
      decision.action.push_back(
          candidate::encode_direction(state.ball, move.to));
      state = ps::apply_move(state, move);
    }
    decision.nodes = search.stats().nodes;
    decision.completed_depth = search.stats().completed_turn_depth;
    decision.root_score = search.stats().root_score;
    decision.neural_evaluations = search.stats().neural_evaluations;
    decision.policy_orderings = search.stats().neural_policy_orderings;
  }
  verify_complete_action(before, state, decision.action);
  return decision;
}

Opening reconstruct(std::string_view transcript, std::string label) {
  Opening opening{ps::make_initial_state(codingame_rules()),
                  std::string(transcript), std::move(label)};
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t separator = transcript.find('/', begin);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    rank5::apply_encoded_turn(opening.state,
                              transcript.substr(begin, end - begin));
    if (separator == std::string_view::npos) {
      break;
    }
    begin = separator + 1U;
  }
  if (ps::is_terminal(opening.state)) {
    throw std::logic_error("curated opening is terminal");
  }
  return opening;
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

Opening random_opening(std::uint64_t seed, int opening_turns,
                       std::string label) {
  for (std::uint64_t attempt = 0; attempt < 10'000; ++attempt) {
    std::uint64_t random = seed + attempt * 0x9e3779b97f4a7c15ULL;
    if (random == 0) {
      random = 1;
    }
    Opening opening{ps::make_initial_state(codingame_rules()), "", label};
    bool valid = true;
    for (int turn = 0; turn < opening_turns && valid; ++turn) {
      const ps::Player mover = opening.state.to_move;
      std::string action;
      while (!ps::is_terminal(opening.state) &&
             opening.state.to_move == mover) {
        const std::vector<ps::Move> moves = ps::legal_moves(opening.state);
        if (moves.empty()) {
          valid = false;
          break;
        }
        const ps::Move move =
            moves[next_random(random) % moves.size()];
        action.push_back(rank5::encode_direction(opening.state.ball, move.to));
        opening.state = ps::apply_move(opening.state, move);
      }
      if (action.empty() || ps::is_terminal(opening.state)) {
        valid = false;
        break;
      }
      append_turn(opening.transcript, action);
    }
    if (valid && !ps::is_terminal(opening.state)) {
      return opening;
    }
  }
  throw std::runtime_error("could not generate a non-terminal opening");
}

GameResult play(const Opening &opening, int candidate_player,
                std::uint64_t node_budget, int maximum_turns) {
  ps::GameState state = opening.state;
  std::string transcript = opening.transcript;
  GameResult result;
  while (!ps::is_terminal(state) && result.turns < maximum_turns) {
    const bool candidate_turn = player_id(state.to_move) == candidate_player;
    const Decision decision =
        candidate_turn
            ? choose_candidate(state, transcript, node_budget, game_time_ms)
                       : choose_rank5(state, transcript, node_budget,
                                      reference_replay_corrections,
                                      game_time_ms);
    append_turn(transcript, decision.action);
    if (candidate_turn) {
      result.candidate_nodes += decision.nodes;
      result.candidate_neural_evaluations += decision.neural_evaluations;
    } else {
      result.reference_nodes += decision.nodes;
    }
    ++result.turns;
  }
  if (const std::optional<ps::Player> winning = ps::winner(state)) {
    result.winner = player_id(*winning);
  }
  return result;
}

void record_pair(const Opening &opening, std::uint64_t node_budget,
                 int maximum_turns, Summary &summary) {
  std::array<GameResult, 2> games{
      play(opening, 0, node_budget, maximum_turns),
      play(opening, 1, node_budget, maximum_turns),
  };
  std::cout << "opening=" << opening.label;
  for (int candidate_player = 0; candidate_player < 2; ++candidate_player) {
    const GameResult &game = games[candidate_player];
    std::cout << " c" << candidate_player << "=" << game.winner
              << ":" << game.turns << ":"
              << game.candidate_neural_evaluations;
    if (game.winner < 0) {
      ++summary.unfinished;
    } else if (game.winner == candidate_player) {
      ++summary.candidate_wins;
      if (candidate_player == 0) {
        ++summary.candidate_as_zero_wins;
      } else {
        ++summary.candidate_as_one_wins;
      }
    } else {
      ++summary.reference_wins;
    }
  }
  std::cout << '\n';
}

void compare_position(std::string_view transcript, std::string_view label,
                      std::uint64_t node_budget,
                      std::uint32_t time_budget_ms,
                      bool replay_corrections = false) {
  const Opening opening = reconstruct(transcript, std::string(label));
  ps::GameState reference_state = opening.state;
  ps::GameState candidate_state = opening.state;
  const Decision reference =
      choose_rank5(reference_state, transcript, node_budget,
                   replay_corrections,
                   time_budget_ms);
  const Decision proposed =
      choose_candidate(candidate_state, transcript, node_budget,
                       time_budget_ms);
  std::cout << "position=" << label << " ball=" << opening.state.ball.x << ','
            << opening.state.ball.y << " mover="
            << player_id(opening.state.to_move) << " rank5="
            << reference.action << ':'
            << reference.completed_depth << ':' << reference.nodes << ':'
            << reference.root_score << ':' << reference.milliseconds
            << " candidate=" << proposed.action << ':'
            << proposed.completed_depth << ':' << proposed.nodes << ':'
            << proposed.root_score << ':' << proposed.milliseconds << ':'
            << proposed.neural_evaluations << ':'
            << proposed.policy_orderings
            << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  if (const char *value = std::getenv("SELFPLAY_VALUE_BLEND")) {
    candidate_value_blend = std::atoi(value);
  }
  if (const char *value = std::getenv("SELFPLAY_POLICY_WEIGHT")) {
    candidate_policy_weight = std::atoi(value);
  }
  if (const char *value = std::getenv("SELFPLAY_GAME_TIME_MS")) {
    game_time_ms = static_cast<std::uint32_t>(std::atoi(value));
  }
  if (const char *value = std::getenv("SELFPLAY_REFERENCE_REPLAY")) {
    reference_replay_corrections = std::atoi(value) != 0;
  }
  if (const char *value = std::getenv("SELFPLAY_CANDIDATE_REPLAY")) {
    candidate_replay_corrections = std::atoi(value) != 0;
  }
  const int pairs_per_phase = argc > 1 ? std::atoi(argv[1]) : 8;
  const std::uint64_t node_budget =
      argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 30'000;
  const int maximum_turns = argc > 3 ? std::atoi(argv[3]) : 200;
  const std::uint32_t position_time_ms =
      argc > 4 ? static_cast<std::uint32_t>(std::atoi(argv[4])) : 0U;
  if (pairs_per_phase < 0 || node_budget == 0 || maximum_turns <= 0) {
    std::cerr << "invalid comparison gate limits\n";
    return 1;
  }

  try {
    Summary summary;
    if (argc > 5) {
      compare_position(argv[5], "command-line", node_budget,
                       position_time_ms, true);
      return 0;
    }
    const std::array<std::pair<std::string_view, std::string_view>, 9>
        positions{{
            {"0", "opening-0-reply"},
            {"0/0", "family-0-0-decision"},
            {"0/0/3", "family-0-0-3-decision"},
            {"0/6/5", "family-0-6-5-decision"},
            {"0/6/5/4/5/53/61/0633", "deltaspace-0633"},
            {"0/6/5/4/5/53/61/431", "deltaspace-431"},
            {"0/6/5/5", "deltaspace-5"},
            {"1/1/7/6/0/75/74/3/003", "deltaspace-player1-003"},
            {"0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/5330",
             "jacek-5330"},
    }};
    for (const auto &[transcript, label] : positions) {
      compare_position(transcript, label, node_budget, position_time_ms);
    }
    if (pairs_per_phase == 0) {
      return 0;
    }

    const std::array<std::pair<std::string_view, std::string_view>, 5>
        curated{{
            {"0/0", "family-0-0"},
            {"0/0/1", "family-0-0-1"},
            {"0/0/3", "family-0-0-3"},
            {"0/6/5", "family-0-6-5"},
            {"0/6/5/4", "family-0-6-5-4"},
        }};
    for (const auto &[transcript, label] : curated) {
      record_pair(reconstruct(transcript, std::string(label)), node_budget,
                  maximum_turns, summary);
    }

    constexpr std::array<int, 4> kOpeningTurns{{2, 6, 12, 20}};
    for (const int turns : kOpeningTurns) {
      for (int pair = 0; pair < pairs_per_phase; ++pair) {
        const std::uint64_t seed =
            0x8f3f73b5cf1c9d6bULL ^
            (static_cast<std::uint64_t>(turns) << 40U) ^
            static_cast<std::uint64_t>(pair + 1);
        const std::string label =
            "random-" + std::to_string(turns) + "-" + std::to_string(pair);
        record_pair(random_opening(seed, turns, label), node_budget,
                    maximum_turns, summary);
      }
    }

    const int games = summary.candidate_wins + summary.reference_wins +
                      summary.unfinished;
    std::cout << "summary games=" << games
              << " candidate=" << summary.candidate_wins
              << " reference=" << summary.reference_wins
              << " unfinished=" << summary.unfinished
              << " candidate_color0=" << summary.candidate_as_zero_wins
              << " candidate_color1=" << summary.candidate_as_one_wins
              << " value_blend=" << candidate_value_blend
              << " policy_weight=" << candidate_policy_weight
              << " game_time_ms=" << game_time_ms
              << " candidate_replay=" << candidate_replay_corrections
              << " reference_replay=" << reference_replay_corrections
              << " node_budget=" << node_budget << '\n';
    return summary.unfinished == 0 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
