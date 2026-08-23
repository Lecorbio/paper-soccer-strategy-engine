#include "jacek_replay_continuations_internal.hpp"

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace continuation = papersoccer::jacek_replay_continuations;

namespace {

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

void require_quotas(std::size_t games,
                    const continuation::ActorQuotas &expected) {
  const continuation::ActorQuotas actual =
      continuation::planned_quotas(1, games);
  require(actual == expected, "unexpected proportional actor quotas");
  std::size_t total = 0;
  for (const std::size_t count : actual) total += count;
  require(total == games, "actor quotas do not sum to requested games");
}

}  // namespace

int main() {
  try {
    require(continuation::planned_quotas(0, 10'000) ==
                continuation::ActorQuotas{{10'000, 0, 0, 0}},
            "canonical round-zero quota changed");
    require(continuation::planned_quotas(1, 10'000) ==
                continuation::ActorQuotas{{0, 5'000, 2'500, 2'500}},
            "canonical round-one quota changed");
    require(continuation::planned_quotas(2, 10'000) ==
                continuation::ActorQuotas{{0, 5'000, 2'500, 2'500}},
            "canonical round-two quota changed");

    require_quotas(1, {{0, 1, 0, 0}});
    require_quotas(2, {{0, 1, 1, 0}});
    require_quotas(3, {{0, 1, 1, 1}});
    require_quotas(4, {{0, 2, 1, 1}});
    require_quotas(5, {{0, 3, 1, 1}});
    require_quotas(6, {{0, 3, 2, 1}});
    require_quotas(7, {{0, 3, 2, 2}});

    bool rejected = false;
    try {
      static_cast<void>(continuation::planned_quotas(3, 1));
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    require(rejected, "invalid round was accepted");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "jacek replay continuation contract test: " << error.what()
              << '\n';
    return 1;
  }
}
