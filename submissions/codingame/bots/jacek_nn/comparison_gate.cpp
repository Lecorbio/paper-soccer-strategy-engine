#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 rank5_engine
#include "../rank_5/bot.cpp"
#undef turn_action_v2

#include "arena_loss_regressions.hpp"
#include "chronological_arena_loss_regressions.hpp"
#include "observed_arena_loss_regressions.hpp"

#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#define turn_action_v2 candidate_engine
#include "bot.cpp"
#undef turn_action_v2

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
namespace candidate = papersoccer::candidate_engine;
namespace regressions = papersoccer::selfplay_nn_v2_regressions;
namespace observed = papersoccer::observed_selfplay_nn_v2_regressions;
namespace chronological = papersoccer::jacek_nn_chronological_regressions;

namespace {

int candidate_residual_weight = 50;
int reference_residual_weight = 0;
std::uint32_t candidate_first_time_ms = 0;
std::uint32_t candidate_later_time_ms = 0;
std::uint32_t reference_first_time_ms = 0;
std::uint32_t reference_later_time_ms = 0;
bool reference_uses_v2 = false;
bool reference_replay_corrections = true;
bool candidate_replay_corrections = true;
bool skip_regression_positions = false;
std::uint32_t candidate_max_depth = 0;
std::uint32_t reference_max_depth = 0;

struct Decision {
  std::string action;
  std::uint64_t nodes{};
  std::uint32_t completed_depth{};
  std::uint32_t attempted_depth{};
  int root_score{};
  double milliseconds{};
  std::uint64_t residual_evaluations{};
  bool budget_exhausted{};
  bool searched{};
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
  std::uint64_t candidate_residual_evaluations{};
  std::uint64_t reference_residual_evaluations{};
  std::uint64_t candidate_completed_depth{};
  std::uint64_t reference_completed_depth{};
  std::uint64_t candidate_attempted_depth{};
  std::uint64_t reference_attempted_depth{};
  std::uint64_t candidate_searches{};
  std::uint64_t reference_searches{};
  std::uint64_t candidate_exhaustions{};
  std::uint64_t reference_exhaustions{};
  double candidate_milliseconds{};
  double reference_milliseconds{};
};

struct Summary {
  int candidate_wins{};
  int reference_wins{};
  int unfinished{};
  int candidate_as_zero_wins{};
  int candidate_as_one_wins{};
  std::uint64_t candidate_nodes{};
  std::uint64_t reference_nodes{};
  std::uint64_t candidate_residual_evaluations{};
  std::uint64_t reference_residual_evaluations{};
  std::uint64_t candidate_completed_depth{};
  std::uint64_t reference_completed_depth{};
  std::uint64_t candidate_attempted_depth{};
  std::uint64_t reference_attempted_depth{};
  std::uint64_t candidate_searches{};
  std::uint64_t reference_searches{};
  std::uint64_t candidate_exhaustions{};
  std::uint64_t reference_exhaustions{};
  double candidate_milliseconds{};
  double reference_milliseconds{};
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
  const auto response_started = rank5::SearchClock::now();
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
    decision.attempted_depth = search.stats().attempted_turn_depth;
    decision.root_score = search.stats().root_score;
    decision.budget_exhausted = search.stats().budget_exhausted;
    decision.searched = true;
  }
  verify_complete_action(before, state, decision.action);
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(
          rank5::SearchClock::now() - response_started)
          .count();
  return decision;
}

Decision choose_candidate(ps::GameState &state, std::string_view transcript,
                          std::uint64_t node_budget,
                          std::uint32_t time_budget_ms = 0,
                          int residual_weight = -1,
                          bool replay_corrections = true,
                          std::uint32_t max_depth = 0) {
  const ps::GameState before = state;
  const auto response_started = candidate::SearchClock::now();
  Decision decision;
  if (!replay_corrections ||
      !candidate::try_replay_correction(state, player_id(state.to_move),
                                        transcript, decision.action)) {
    candidate::SearchConfig config;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent =
        residual_weight < 0 ? candidate_residual_weight : residual_weight;
    if (max_depth != 0) {
      config.max_turn_depth = max_depth;
    }
    const auto started = candidate::SearchClock::now();
    if (time_budget_ms == 0) {
      config.max_nodes = node_budget;
    } else {
      config.absolute_deadline =
          started + std::chrono::milliseconds(time_budget_ms);
    }
    candidate::CompleteTurnSearch search(state, config);
    const std::vector<ps::Move> moves = search.run();
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
    decision.attempted_depth = search.stats().attempted_turn_depth;
    decision.root_score = search.stats().root_score;
    decision.residual_evaluations =
        search.stats().teacher_residual_evaluations;
    decision.budget_exhausted = search.stats().budget_exhausted;
    decision.searched = true;
  }
  verify_complete_action(before, state, decision.action);
  decision.milliseconds =
      std::chrono::duration<double, std::milli>(
          candidate::SearchClock::now() - response_started)
          .count();
  return decision;
}

Decision choose_reference(ps::GameState &state, std::string_view transcript,
                          std::uint64_t node_budget,
                          std::uint32_t time_budget_ms) {
  if (reference_uses_v2) {
    return choose_candidate(state, transcript, node_budget, time_budget_ms,
                            reference_residual_weight,
                            reference_replay_corrections,
                            reference_max_depth);
  }
  return choose_rank5(state, transcript, node_budget,
                      reference_replay_corrections, time_budget_ms);
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
  const std::uint32_t prior_turns = opening.transcript.empty()
                                        ? 0U
                                        : 1U + static_cast<std::uint32_t>(
                                                   std::count(
                                                       opening.transcript.begin(),
                                                       opening.transcript.end(),
                                                       '/'));
  const std::array<std::uint32_t, 2> prior_player_turns{
      (prior_turns + 1U) / 2U, prior_turns / 2U};
  std::uint32_t candidate_turns = prior_player_turns[candidate_player];
  std::uint32_t reference_turns = prior_player_turns[1 - candidate_player];
  while (!ps::is_terminal(state) && result.turns < maximum_turns) {
    const bool candidate_turn = player_id(state.to_move) == candidate_player;
    const std::uint32_t time_budget =
        candidate_turn
            ? (candidate_turns == 0 ? candidate_first_time_ms
                                    : candidate_later_time_ms)
            : (reference_turns == 0 ? reference_first_time_ms
                                    : reference_later_time_ms);
    const Decision decision =
        candidate_turn
            ? choose_candidate(state, transcript, node_budget, time_budget,
                               candidate_residual_weight,
                               candidate_replay_corrections,
                               candidate_max_depth)
            : choose_reference(state, transcript, node_budget, time_budget);
    append_turn(transcript, decision.action);
    if (candidate_turn) {
      ++candidate_turns;
      result.candidate_nodes += decision.nodes;
      result.candidate_residual_evaluations += decision.residual_evaluations;
      result.candidate_completed_depth += decision.completed_depth;
      result.candidate_attempted_depth += decision.attempted_depth;
      result.candidate_milliseconds += decision.milliseconds;
      result.candidate_searches += decision.searched ? 1U : 0U;
      result.candidate_exhaustions += decision.budget_exhausted ? 1U : 0U;
    } else {
      ++reference_turns;
      result.reference_nodes += decision.nodes;
      result.reference_residual_evaluations +=
          decision.residual_evaluations;
      result.reference_completed_depth += decision.completed_depth;
      result.reference_attempted_depth += decision.attempted_depth;
      result.reference_milliseconds += decision.milliseconds;
      result.reference_searches += decision.searched ? 1U : 0U;
      result.reference_exhaustions += decision.budget_exhausted ? 1U : 0U;
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
              << ':' << game.turns << ':' << game.candidate_nodes
              << ':' << game.reference_nodes << ':'
              << game.candidate_completed_depth << '/'
              << game.candidate_searches << ':'
              << game.reference_completed_depth << '/'
              << game.reference_searches << ':'
              << game.candidate_exhaustions << '/'
              << game.reference_exhaustions;
    summary.candidate_nodes += game.candidate_nodes;
    summary.reference_nodes += game.reference_nodes;
    summary.candidate_residual_evaluations +=
        game.candidate_residual_evaluations;
    summary.reference_residual_evaluations +=
        game.reference_residual_evaluations;
    summary.candidate_completed_depth += game.candidate_completed_depth;
    summary.reference_completed_depth += game.reference_completed_depth;
    summary.candidate_attempted_depth += game.candidate_attempted_depth;
    summary.reference_attempted_depth += game.reference_attempted_depth;
    summary.candidate_searches += game.candidate_searches;
    summary.reference_searches += game.reference_searches;
    summary.candidate_exhaustions += game.candidate_exhaustions;
    summary.reference_exhaustions += game.reference_exhaustions;
    summary.candidate_milliseconds += game.candidate_milliseconds;
    summary.reference_milliseconds += game.reference_milliseconds;
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
                      std::uint32_t time_budget_ms) {
  const Opening opening = reconstruct(transcript, std::string(label));
  const std::uint32_t prior_turns = transcript.empty()
                                        ? 0U
                                        : 1U + static_cast<std::uint32_t>(
                                                   std::count(transcript.begin(),
                                                              transcript.end(),
                                                              '/'));
  const std::uint32_t mover_turns =
      player_id(opening.state.to_move) == 0 ? (prior_turns + 1U) / 2U
                                            : prior_turns / 2U;
  const std::uint32_t candidate_time =
      time_budget_ms != 0
          ? time_budget_ms
          : (mover_turns == 0 ? candidate_first_time_ms
                              : candidate_later_time_ms);
  const std::uint32_t reference_time =
      time_budget_ms != 0
          ? time_budget_ms
          : (mover_turns == 0 ? reference_first_time_ms
                              : reference_later_time_ms);
  ps::GameState reference_state = opening.state;
  ps::GameState candidate_state = opening.state;
  const Decision reference =
      reference_uses_v2
          ? choose_candidate(reference_state, transcript, node_budget,
                             reference_time, reference_residual_weight,
                             reference_replay_corrections,
                             reference_max_depth)
          : choose_rank5(reference_state, transcript, node_budget,
                         reference_replay_corrections, reference_time);
  const Decision proposed =
      choose_candidate(candidate_state, transcript, node_budget,
                       candidate_time, candidate_residual_weight,
                       candidate_replay_corrections,
                       candidate_max_depth);
  std::cout << "position=" << label << " ball=" << opening.state.ball.x << ','
            << opening.state.ball.y << " mover="
            << player_id(opening.state.to_move) << " reference="
            << (reference_uses_v2 ? "jacek_nn:" : "rank_5:")
            << reference.action << ':'
            << reference.completed_depth << '/' << reference.attempted_depth
            << ':' << reference.nodes << ':'
            << reference.root_score << ':' << reference.milliseconds << ':'
            << reference.residual_evaluations << ':'
            << reference.budget_exhausted
            << " candidate=" << proposed.action << ':'
            << proposed.completed_depth << '/' << proposed.attempted_depth
            << ':' << proposed.nodes << ':'
            << proposed.root_score << ':' << proposed.milliseconds << ':'
            << proposed.residual_evaluations << ':'
            << proposed.budget_exhausted
            << " changed=" << (reference.action != proposed.action)
            << '\n';
}

}  // namespace

int main(int argc, char **argv) {
  if (const char *value = std::getenv("JACEK_NN_RESIDUAL_WEIGHT")) {
    candidate_residual_weight = std::atoi(value);
  }
  if (const char *value =
          std::getenv("JACEK_NN_REFERENCE_RESIDUAL_WEIGHT")) {
    reference_residual_weight = std::atoi(value);
  }
  if (const char *value = std::getenv("JACEK_NN_GAME_TIME_MS")) {
    const std::uint32_t time =
        static_cast<std::uint32_t>(std::atoi(value));
    candidate_first_time_ms = candidate_later_time_ms = time;
    reference_first_time_ms = reference_later_time_ms = time;
  }
  if (const char *value = std::getenv("JACEK_NN_CANDIDATE_FIRST_MS")) {
    candidate_first_time_ms =
        static_cast<std::uint32_t>(std::atoi(value));
  }
  if (const char *value = std::getenv("JACEK_NN_CANDIDATE_LATER_MS")) {
    candidate_later_time_ms =
        static_cast<std::uint32_t>(std::atoi(value));
  }
  if (const char *value = std::getenv("JACEK_NN_REFERENCE_FIRST_MS")) {
    reference_first_time_ms =
        static_cast<std::uint32_t>(std::atoi(value));
  }
  if (const char *value = std::getenv("JACEK_NN_REFERENCE_LATER_MS")) {
    reference_later_time_ms =
        static_cast<std::uint32_t>(std::atoi(value));
  }
  if (const char *value = std::getenv("JACEK_NN_REFERENCE_ENGINE")) {
    reference_uses_v2 = std::atoi(value) != 0;
  }
  if (const char *value = std::getenv("JACEK_NN_REFERENCE_REPLAY")) {
    reference_replay_corrections = std::atoi(value) != 0;
  }
  if (const char *value = std::getenv("JACEK_NN_CANDIDATE_REPLAY")) {
    candidate_replay_corrections = std::atoi(value) != 0;
  }
  if (const char *value = std::getenv("JACEK_NN_SKIP_REGRESSIONS")) {
    skip_regression_positions = std::atoi(value) != 0;
  }
  if (const char *value = std::getenv("JACEK_NN_MAX_DEPTH")) {
    candidate_max_depth = static_cast<std::uint32_t>(std::atoi(value));
  }
  if (const char *value =
          std::getenv("JACEK_NN_REFERENCE_MAX_DEPTH")) {
    reference_max_depth = static_cast<std::uint32_t>(std::atoi(value));
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
                       position_time_ms);
      return 0;
    }
    const auto compare_regression = [&](const auto &position) {
      const Opening verified = reconstruct(position.prefix,
                                           std::string(position.label));
      if (player_id(verified.state.to_move) != position.player_id) {
        throw std::logic_error("arena-loss regression player mismatch");
      }
      compare_position(position.prefix, position.label, node_budget,
                       position_time_ms);
    };
    if (!skip_regression_positions) {
      for (const regressions::Case &position : regressions::kCases) {
        compare_regression(position);
      }
      for (const observed::Case &position : observed::kCases) {
        compare_regression(position);
      }
      for (const chronological::Case &position : chronological::kCases) {
        compare_regression(position);
      }
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
    const double candidate_average_depth =
        summary.candidate_searches == 0
            ? 0.0
            : static_cast<double>(summary.candidate_completed_depth) /
                  summary.candidate_searches;
    const double reference_average_depth =
        summary.reference_searches == 0
            ? 0.0
            : static_cast<double>(summary.reference_completed_depth) /
                  summary.reference_searches;
    const double candidate_average_attempted =
        summary.candidate_searches == 0
            ? 0.0
            : static_cast<double>(summary.candidate_attempted_depth) /
                  summary.candidate_searches;
    const double reference_average_attempted =
        summary.reference_searches == 0
            ? 0.0
            : static_cast<double>(summary.reference_attempted_depth) /
                  summary.reference_searches;
    std::cout << "summary games=" << games
              << " candidate=" << summary.candidate_wins
              << " reference=" << summary.reference_wins
              << " unfinished=" << summary.unfinished
              << " candidate_color0=" << summary.candidate_as_zero_wins
              << " candidate_color1=" << summary.candidate_as_one_wins
              << " candidate_nodes=" << summary.candidate_nodes
              << " reference_nodes=" << summary.reference_nodes
              << " candidate_depth=" << candidate_average_depth
              << " reference_depth=" << reference_average_depth
              << " candidate_attempted=" << candidate_average_attempted
              << " reference_attempted=" << reference_average_attempted
              << " candidate_exhaustions=" << summary.candidate_exhaustions
              << " reference_exhaustions=" << summary.reference_exhaustions
              << " candidate_ms=" << summary.candidate_milliseconds
              << " reference_ms=" << summary.reference_milliseconds
              << " residual_evaluations="
              << summary.candidate_residual_evaluations
              << " reference_residual_evaluations="
              << summary.reference_residual_evaluations
              << " residual_weight=" << candidate_residual_weight
              << " reference_residual_weight=" << reference_residual_weight
              << " reference_engine="
              << (reference_uses_v2 ? "jacek_nn" : "rank_5")
              << " candidate_time=" << candidate_first_time_ms << '/'
              << candidate_later_time_ms
              << " reference_time=" << reference_first_time_ms << '/'
              << reference_later_time_ms
              << " candidate_replay=" << candidate_replay_corrections
              << " reference_replay=" << reference_replay_corrections
              << " max_depth=" << candidate_max_depth
              << " reference_max_depth=" << reference_max_depth
              << " node_budget=" << node_budget << '\n';
    return summary.unfinished == 0 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
