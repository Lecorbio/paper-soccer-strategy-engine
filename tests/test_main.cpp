int run_rules_tests();
int run_bot_tests();

int main() {
  const int rules_failures = run_rules_tests();
  const int bot_failures = run_bot_tests();
  return (rules_failures == 0 && bot_failures == 0) ? 0 : 1;
}
