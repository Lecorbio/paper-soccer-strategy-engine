int run_rules_tests();
int run_bot_tests();
int run_match_tests();
int run_mcts_tests();
int run_web_game_session_tests();

int main() {
  const int rules_failures = run_rules_tests();
  const int bot_failures = run_bot_tests();
  const int match_failures = run_match_tests();
  const int mcts_failures = run_mcts_tests();
  const int web_game_session_failures = run_web_game_session_tests();
  return (rules_failures == 0 && bot_failures == 0 && match_failures == 0 &&
          mcts_failures == 0 && web_game_session_failures == 0)
             ? 0
             : 1;
}
