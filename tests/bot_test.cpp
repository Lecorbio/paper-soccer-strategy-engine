#include <iostream>
#include <stdexcept>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

bool contains_move(const std::vector<ps::Move> &moves, ps::Point point) {
  for (const ps::Move &move : moves) {
    if (move.to == point) {
      return true;
    }
  }
  return false;
}

std::vector<ps::Point> play_random_game(std::uint64_t player_one_seed,
                                        std::uint64_t player_two_seed) {
  ps::RandomBot player_one_bot(player_one_seed);
  ps::RandomBot player_two_bot(player_two_seed);
  ps::GameState state = ps::make_initial_state();

  std::size_t plies = 0;
  while (!ps::is_terminal(state)) {
    const std::vector<ps::Move> legal = ps::legal_moves(state);
    require(!legal.empty(), "Test game should not reach a blocked non-terminal state.");

    ps::Bot &bot =
        (state.to_move == ps::Player::One) ? static_cast<ps::Bot &>(player_one_bot)
                                           : static_cast<ps::Bot &>(player_two_bot);
    const ps::Move move = bot.choose_move(state);
    require(contains_move(legal, move.to), "RandomBot must choose a legal move.");

    state = ps::apply_move(state, move);
    ++plies;
    require(plies <= 256, "Random self-play should terminate in a finite number of plies.");
  }

  return state.path;
}

void random_bot_chooses_a_legal_move() {
  ps::RandomBot bot(7);
  const ps::GameState state = ps::make_initial_state();
  const ps::Move move = bot.choose_move(state);
  require(contains_move(ps::legal_moves(state), move.to),
          "RandomBot should always return one of the legal moves.");
}

void random_bot_is_deterministic_for_same_seed_pair() {
  const std::vector<ps::Point> first = play_random_game(17, 23);
  const std::vector<ps::Point> second = play_random_game(17, 23);
  require(first == second, "The same seed pair should replay the same self-play game.");
}

void random_bot_rejects_states_without_legal_moves() {
  ps::RandomBot bot(5);
  ps::GameState state = ps::make_initial_state();
  state.status = ps::Status::WonByOne;

  bool threw = false;
  try {
    (void)bot.choose_move(state);
  } catch (const std::invalid_argument &) {
    threw = true;
  }

  require(threw, "RandomBot must reject terminal states.");
}

}  // namespace

int run_bot_tests() {
  struct TestCase {
    const char *name;
    void (*run)();
  };

  const std::vector<TestCase> tests{
      {"random_bot_chooses_a_legal_move", random_bot_chooses_a_legal_move},
      {"random_bot_is_deterministic_for_same_seed_pair",
       random_bot_is_deterministic_for_same_seed_pair},
      {"random_bot_rejects_states_without_legal_moves",
       random_bot_rejects_states_without_legal_moves},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << "\n";
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << "\n";
    } catch (...) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": unknown error\n";
    }
  }

  std::cout << "\n" << (tests.size() - static_cast<std::size_t>(failures)) << "/"
            << tests.size() << " bot tests passed.\n";
  return failures == 0 ? 0 : 1;
}
