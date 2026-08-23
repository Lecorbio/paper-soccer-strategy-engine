int run_rules_tests();
int run_game_review_tests();
int run_bot_tests();
int run_alpha_beta_tests();
int run_jacek_inspired_tests();
int run_jacek_replay_bfm_tests();
int run_match_tests();
int run_mcts_tests();
int run_rank5_derived_tests();
int run_web_game_session_tests();

int main() {
  const int game_review_failures = run_game_review_tests();
  const int rules_failures = run_rules_tests();
  const int bot_failures = run_bot_tests();
  const int alpha_beta_failures = run_alpha_beta_tests();
  const int jacek_inspired_failures = run_jacek_inspired_tests();
  const int jacek_replay_bfm_failures = run_jacek_replay_bfm_tests();
  const int match_failures = run_match_tests();
  const int mcts_failures = run_mcts_tests();
  const int rank5_derived_failures = run_rank5_derived_tests();
  const int web_game_session_failures = run_web_game_session_tests();
  return (game_review_failures == 0 && rules_failures == 0 &&
          bot_failures == 0 && alpha_beta_failures == 0 &&
          jacek_inspired_failures == 0 && match_failures == 0 &&
          jacek_replay_bfm_failures == 0 &&
          mcts_failures == 0 && rank5_derived_failures == 0 &&
          web_game_session_failures == 0)
             ? 0
             : 1;
}
