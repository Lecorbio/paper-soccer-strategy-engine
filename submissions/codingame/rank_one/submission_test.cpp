#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "paper_soccer_rank_one.cpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

constexpr std::string_view kSelectedTranscript =
    "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474";
constexpr std::string_view kSelectedAction = "42474176";

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_invalid_argument(Function &&function, const std::string &message) {
  bool threw = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  require(threw, message);
}

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::GameState make_clean_state_at(ps::Point point,
                                  ps::Player player = ps::Player::One) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  state.ball = point;
  state.to_move = player;
  state.status = ps::Status::InProgress;
  state.path = {point};
  state.used_segments.clear();
  state.visit_count.clear();
  state.visit_count[point] = 1;
  return state;
}

bool same_state(const ps::GameState &left, const ps::GameState &right) {
  return left.config.width == right.config.width &&
         left.config.height == right.config.height &&
         left.config.goal_rule == right.config.goal_rule &&
         left.config.blocked_rule == right.config.blocked_rule &&
         left.ball == right.ball && left.to_move == right.to_move &&
         left.status == right.status && left.path == right.path &&
         left.used_segments == right.used_segments &&
         left.visit_count == right.visit_count;
}

void block_edges_except(ps::GameState &state, ps::Point from,
                        const std::vector<ps::Point> &allowed) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const ps::Point destination{from.x + dx, from.y + dy};
      if (std::find(allowed.begin(), allowed.end(), destination) ==
          allowed.end()) {
        state.used_segments.insert(ps::Segment{from, destination});
      }
    }
  }
}

ps::GameState forced_two_move_rebound() {
  const ps::Point root{4, 4};
  const ps::Point rebound{4, 3};
  const ps::Point handoff{4, 2};
  ps::GameState state = make_clean_state_at(root);
  state.visit_count[rebound] = 1;
  block_edges_except(state, root, {rebound});
  block_edges_except(state, rebound, {root, handoff});
  return state;
}

std::vector<std::string_view> split_transcript(std::string_view transcript) {
  std::vector<std::string_view> turns;
  std::size_t start = 0;
  while (start <= transcript.size()) {
    const std::size_t separator = transcript.find('/', start);
    const std::size_t end = separator == std::string_view::npos
                                ? transcript.size()
                                : separator;
    require(end != start, "Transcript contains an empty turn.");
    turns.push_back(transcript.substr(start, end - start));
    if (separator == std::string_view::npos) {
      break;
    }
    start = separator + 1;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(codingame_rules());
  for (const std::string_view turn : split_transcript(transcript)) {
    cg::apply_encoded_turn(state, turn);
  }
  return state;
}

void directions_round_trip_exactly() {
  const ps::Point origin{4, 6};
  constexpr std::array<ps::Point, 8> expected{{
      {4, 5}, {5, 5}, {5, 6}, {5, 7},
      {4, 7}, {3, 7}, {3, 6}, {3, 5},
  }};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const char direction = static_cast<char>('0' + index);
    require(cg::decode_direction(origin, direction).to == expected[index],
            "Direction decoding should match the CodinGame compass table.");
    require(cg::encode_direction(origin, expected[index]) == direction,
            "Direction encoding should invert every valid direction.");
  }
  require_invalid_argument(
      [&] { (void)cg::decode_direction(origin, '8'); },
      "Direction eight must never be accepted or emitted.");
}

void own_goals_and_blocked_mover_follow_contest_rules() {
  ps::GameState north = make_clean_state_at({4, 1}, ps::Player::Two);
  north = ps::apply_move(north, {{4, 0}});
  require(ps::winner(north) == ps::Player::One,
          "Player Two's north own goal should award Player One.");

  ps::GameState south = make_clean_state_at({4, 11}, ps::Player::One);
  south = ps::apply_move(south, {{4, 12}});
  require(ps::winner(south) == ps::Player::Two,
          "Player One's south own goal should award Player Two.");

  ps::GameState blocked = make_clean_state_at({4, 6}, ps::Player::One);
  block_edges_except(blocked, {4, 5}, {{4, 6}});
  blocked = ps::apply_move(blocked, {{4, 5}});
  require(ps::winner(blocked) == ps::Player::Two,
          "A blocked landing should immediately defeat the mover.");
}

void encoded_turn_validation_is_atomic() {
  ps::GameState rebound = forced_two_move_rebound();
  const ps::GameState snapshot = rebound;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(rebound, "0"); },
      "A turn may not stop during a mandatory rebound.");
  require(same_state(rebound, snapshot),
          "A rejected early turn should not partially mutate state.");

  cg::apply_encoded_turn(rebound, "00");
  require(rebound.ball == ps::Point{4, 2} &&
              rebound.to_move == ps::Player::Two,
          "A complete rebound sequence should hand possession over.");

  ps::GameState initial = ps::make_initial_state(codingame_rules());
  const ps::GameState initial_snapshot = initial;
  require_invalid_argument(
      [&] { cg::apply_encoded_turn(initial, "00"); },
      "A turn may not include an opponent's next move.");
  require(same_state(initial, initial_snapshot),
          "A rejected extra move should not partially mutate state.");
}

void timed_search_returns_a_complete_replayable_action() {
  const ps::GameState original = ps::make_initial_state(codingame_rules());
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(!action.empty(), "Timed search should return an action.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed),
          "The timed action should reconstruct the chosen state.");
}

void normal_fallback_finishes_mandatory_rebounds() {
  const ps::GameState original = forced_two_move_rebound();
  ps::GameState chosen = original;
  const std::string action = cg::choose_complete_turn(chosen, 1);
  require(action == "00",
          "The normal V2 path should finish the forced rebound action.");

  ps::GameState replayed = original;
  cg::apply_encoded_turn(replayed, action);
  require(same_state(chosen, replayed) &&
              replayed.to_move == ps::Player::Two,
          "The fallback action should be legal and rebound-complete.");
}

void interrupted_search_preserves_a_complete_action() {
  const ps::GameState state = forced_two_move_rebound();
  cg::SearchConfig config;
  config.max_nodes = 1;
  config.max_time_ms = 0;
  cg::CompleteTurnSearch search(state, config);
  const std::vector<ps::Move> moves = search.run();
  require(search.stats().budget_exhausted && !moves.empty(),
          "A one-node search should return its complete fallback action.");

  ps::GameState replayed = state;
  std::string encoded;
  for (const ps::Move move : moves) {
    encoded.push_back(cg::encode_direction(replayed.ball, move.to));
    replayed = ps::apply_move(replayed, move);
  }
  ps::GameState verified = state;
  cg::apply_encoded_turn(verified, encoded);
  require(same_state(replayed, verified),
          "The interrupted search fallback should replay exactly.");
}

void selected_replay_correction_activates_exactly() {
  ps::GameState actual = reconstruct(kSelectedTranscript);
  ps::GameState expected = actual;
  cg::apply_encoded_turn(expected, kSelectedAction);

  std::string encoded = "sentinel";
  require(cg::try_replay_correction(actual, 0, kSelectedTranscript, encoded),
          "The independently accepted correction should activate.");
  require(encoded == kSelectedAction,
          "The accepted correction should return the frozen action.");
  require(same_state(actual, expected),
          "The accepted correction should reach the directly replayed state.");
}


void complete_replay_book_is_legal_and_exact() {
  for (const cg::replay_book::Replay &replay : cg::replay_book::kReplays) {
    ps::GameState state = ps::make_initial_state(codingame_rules());
    std::string prefix;
    std::size_t turn = 0;
    std::size_t begin = 0;
    while (begin < replay.transcript.size()) {
      const std::size_t separator = replay.transcript.find('/', begin);
      const std::size_t end = separator == std::string_view::npos
                                  ? replay.transcript.size()
                                  : separator;
      const std::string_view action = replay.transcript.substr(begin, end - begin);
      require(!action.empty(), "A replay-book action cannot be empty.");
      if (turn >= replay.first_turn && turn % 2U == replay.player_id) {
        ps::GameState actual = state;
        ps::GameState expected = state;
        cg::apply_encoded_turn(expected, action);
        std::string encoded = "sentinel";
        require(cg::try_replay_correction(actual, replay.player_id, prefix,
                                          encoded),
                "Every eligible replay-book decision should activate.");
        require(encoded == action && same_state(actual, expected),
                "Every replay-book decision should be exact and legal.");
      }
      cg::apply_encoded_turn(state, action);
      if (!prefix.empty()) {
        prefix.push_back('/');
      }
      prefix.append(action);
      ++turn;
      if (separator == std::string_view::npos) {
        break;
      }
      begin = separator + 1U;
    }
  }
}

void rejected_replay_corrections_are_absent() {
  constexpr std::array<std::string_view, 3> rejected{{
      "7/6/0/5/71/335",
      "7/6/0/5/71/335/01/44/21/2/0/3/17/554",
      "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474/"
      "42474177/65",
  }};
  for (const std::string_view transcript : rejected) {
    ps::GameState state = reconstruct(transcript);
    const ps::GameState before = state;
    std::string encoded = "sentinel";
    require(!cg::try_replay_correction(state, 0, transcript, encoded),
            "An independently rejected correction activated.");
    require(same_state(state, before) && encoded == "sentinel",
            "A rejected correction should preserve state and output.");
  }
}

void replay_correction_fallback_is_safe() {
  ps::GameState wrong_player = reconstruct(kSelectedTranscript);
  const ps::GameState wrong_player_before = wrong_player;
  std::string encoded = "sentinel";
  require(!cg::try_replay_correction(
              wrong_player, 1, kSelectedTranscript, encoded),
          "The correction should not activate for Player 1.");
  require(same_state(wrong_player, wrong_player_before) &&
              encoded == "sentinel",
          "Wrong-player lookup should preserve state and output.");

  ps::GameState unknown = ps::make_initial_state(codingame_rules());
  const ps::GameState unknown_before = unknown;
  require(!cg::try_replay_correction(unknown, 0, "not/a/transcript", encoded),
          "An unknown transcript should not activate.");
  require(same_state(unknown, unknown_before) && encoded == "sentinel",
          "Unknown lookup should preserve state and output.");

  ps::GameState terminal = reconstruct(kSelectedTranscript);
  terminal.status = ps::Status::WonByOne;
  const ps::GameState terminal_before = terminal;
  require(!cg::try_replay_correction(
              terminal, 0, kSelectedTranscript, encoded),
          "An illegal correction should fail its copy-before-apply guard.");
  require(same_state(terminal, terminal_before) && encoded == "sentinel",
          "An illegal correction should preserve state and output.");
}

}  // namespace

int main() {
  struct TestCase {
    const char *name;
    void (*run)();
  };
  const std::vector<TestCase> tests{
      {"directions_round_trip_exactly", directions_round_trip_exactly},
      {"own_goals_and_blocked_mover_follow_contest_rules",
       own_goals_and_blocked_mover_follow_contest_rules},
      {"encoded_turn_validation_is_atomic",
       encoded_turn_validation_is_atomic},
      {"timed_search_returns_a_complete_replayable_action",
       timed_search_returns_a_complete_replayable_action},
      {"normal_fallback_finishes_mandatory_rebounds",
       normal_fallback_finishes_mandatory_rebounds},
      {"interrupted_search_preserves_a_complete_action",
       interrupted_search_preserves_a_complete_action},
      {"selected_replay_correction_activates_exactly",
       selected_replay_correction_activates_exactly},
      {"complete_replay_book_is_legal_and_exact",
       complete_replay_book_is_legal_and_exact},
      {"rejected_replay_corrections_are_absent",
       rejected_replay_corrections_are_absent},
      {"replay_correction_fallback_is_safe",
       replay_correction_fallback_is_safe},
  };

  int failures = 0;
  for (const TestCase &test : tests) {
    try {
      test.run();
      std::cout << "[PASS] " << test.name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cout << "[FAIL] " << test.name << ": " << error.what() << '\n';
    }
  }
  std::cout << '\n' << (tests.size() - static_cast<std::size_t>(failures))
            << '/' << tests.size() << " submission tests passed.\n";
  return failures == 0 ? 0 : 1;
}
