#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "submission.cpp"

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

void elite_replay_paths_are_legal_and_exact() {
  struct Replay {
    int player_id;
    std::size_t first_corrected_turn;
    std::string_view transcript;
  };
  static constexpr std::array<Replay, 12> replays{{
      {1, 3,
       "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35/7/164675/6/6/71/"
       "72524/4611232/764/763/30654/17201/3550102144444/1721/425/46/16467/"
       "5/024674/75252/53/530/1/0/711324/75/0632/3/1/3/17505675/23/"
       "2027545/2/13/3/25050/50/11356675/47/4772420613520633/10/3067425/"
       "46165"},
      {0, 6,
       "0/5/21/2/7/523/3/35/0/36350/07/5/6/6/0/5/5/631/16430/5023/21/"
       "41674/7577/2/21/3/577/55327/724101305/533/3050100/321647/66666/"
       "72/5202/243/11425056542006074772/71/714/3/66144/3/6302/330/007/2/"
       "2/2/0/0363636/7507/13/34546764750014/71/05211"},
      {1, 5,
       "1/1/7/6/6/3/07/05/74/53/1/2060350634/14/663/1211/4/1/03635/"
       "6060/1474553/032/17723/364/27056/56/30206577/724452/22076524144/"
       "2/0574/1313/12744/3/17563/235/350/17/1647074665/3/5/2/01/05/"
       "0367635/61/7241466335/6300/61420/742164203/02/3/24705/652/105633/"
       "6/35"},
      {0, 4,
       "0/6/5/5/53/4/610/063523/503030/60/521/4723/1/4/31/2507/2/46/"
       "054177/50/561/222/3607/05/4147547270/1/46632/221/3/3/360257/"
       "05241656/670771/4/22322/0/3/36/72250507/477/5763341/4721222/17/"
       "0/035275/5/0/5/6412477/"},
      {0, 6,
       "0/1/0/6/33/5/3/6/5/3/6/05/21/66/7/4/6/3/161/13442/72/01/17/"
       "0/255336/5702/3025607/22/520167/5446166/6/1/7/7/0524/53/27/"
       "0255253/2244775272417/2/70721/61444/306541414/502135/4720632100/"
       "161/035074/52424766/050/714/52031/30744357472720/71"},
      {0, 2,
       "0/6/5/4/5/53/61/72442/30/2/7/46413/00/2/531/316/50216/56/52076572/450120"
       "/357177/532/21/03/2/3/2500/65/441/24763072/4167474/2507/2253117/50/24765"
       "6431605470/7272/350230/270/330/1/7/2/42/25/2743527/677/45017/2/46124/520"
       "1317/05/74/53611631757/65/31/30616474/60317"},
      {1, 1,
       "4/6/6/3/6/3/2/746/6/14117/7/7/71/63/003/14/6544/61613/0/7144524612/4/617"
       "13/4/53550143/01/6605/74/20553/6301021/72432/4/2/75/555/7531070/72361342"
       "3/2/4/71/43/17/0522725743/425750/306064/5747/72422/13610/722744641320555"
       "/0603364"},
      {1, 3,
       "3/1/7/7/7/2/2/505/6/6/1/7/5352/7243/2/3/6/02574/5/3/6/6171/6110300/5253/"
       "312/1/7/2/0/25252744/64136/02566/44/71134/4/27254/4/166/74/6/16/323257/5"
       "0/1467016023/03663/534705/07/2050610164563353/57/006353012/3647/23/02525"},
      {1, 3,
       "0/0/3/67/27/45/5/71/1/7/52/0053461231/65/4725/52023/5/4/5/0/175323/0/64/"
       "6616352/111/060/7242533/0/3/2/5/0633/57/7/64/111/3356/50202/02575/6772/2"
       "27557/4321/34160644/505741/33"},
      {0, 40,
       "0/3/2/3/6/5/2/4/76/5/6/4/10/55/2476317/1/221/030524745/674663071/36/071/"
       "3/0/2/21/3/5671/47/5671/4/72714/43/65212221/4/2/505/247021/2506335/2553/"
       "66/677/0174744/366311/1256650134/6413312/005325357/014271761/0270/56/57/"
       "47/22/2470/13/030522575771"},
      {0, 18,
       "0/5/21/3/0/3/17/55/061/4276/1/42/217/4/427570/5/4432225/35/36/57/1/4/125"
       "664130061/02745/2470/274763/25/5/0/5/230/65/57/4721/2470/5/71/335/724247"
       "6061/721/0/364/671/33/0523111/60575/6671/4/721/7533/6301/74/2221/3166614"
       "743/67/756/3552133550500/33/6631231642706/427543/07027027/4561160327417/"
       "01"},
      {1, 3,
       "7/6/1/44/21/7/7/422/7/455/00/21657/1/75353/10/27433/141/3/17/25/42700/25"
       "6/3607/135524/665760/166352325/06/33/0/10676356144/133/3/1/6/1/44/2/71/7"
       "/42355/01/435755"},

  }};

  for (const Replay &replay : replays) {
    ps::GameState state = ps::make_initial_state(codingame_rules());
    std::string prefix;
    std::size_t turn = 0;
    std::size_t begin = 0;
    while (begin <= replay.transcript.size()) {
      const std::size_t separator = replay.transcript.find('/', begin);
      const std::size_t end = separator == std::string_view::npos
                                  ? replay.transcript.size()
                                  : separator;
      const std::string_view action = replay.transcript.substr(begin, end - begin);
      if (action.empty()) {
        break;
      }
      if (turn >= replay.first_corrected_turn &&
          (state.to_move == ps::Player::One ? 0 : 1) == replay.player_id) {
        ps::GameState actual = state;
        ps::GameState expected = state;
        cg::apply_encoded_turn(expected, action);
        std::string encoded = "sentinel";
        require(cg::try_replay_correction(actual, replay.player_id, prefix,
                                          encoded),
                "An elite replay correction should activate.");
        require(encoded == action && same_state(actual, expected),
                "An elite replay correction should apply its exact action.");
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
      begin = separator + 1;
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
      {"elite_replay_paths_are_legal_and_exact",
       elite_replay_paths_are_legal_and_exact},
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
