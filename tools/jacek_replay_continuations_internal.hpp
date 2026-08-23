#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace papersoccer::jacek_replay_continuations {

enum class ActorMode : std::size_t {
  Rank4VsRank4 = 0,
  CandidateSelfplay = 1,
  CandidatePlayerOne = 2,
  CandidatePlayerTwo = 3,
};

using ActorQuotas = std::array<std::size_t, 4>;

inline constexpr std::string_view actor_mode_name(ActorMode mode) noexcept {
  switch (mode) {
    case ActorMode::Rank4VsRank4:
      return "rank4-vs-rank4";
    case ActorMode::CandidateSelfplay:
      return "candidate-selfplay";
    case ActorMode::CandidatePlayerOne:
      return "candidate-p1-vs-rank4";
    case ActorMode::CandidatePlayerTwo:
      return "candidate-p2-vs-rank4";
  }
  return "invalid";
}

inline constexpr std::string_view candidate_color(ActorMode mode) noexcept {
  switch (mode) {
    case ActorMode::Rank4VsRank4:
      return "none";
    case ActorMode::CandidateSelfplay:
      return "both";
    case ActorMode::CandidatePlayerOne:
      return "player-one";
    case ActorMode::CandidatePlayerTwo:
      return "player-two";
  }
  return "invalid";
}

// Development-sized rounds use Hamilton's largest-remainder apportionment of
// the canonical 2:1:1 mix. Equal remainders are resolved in the stable order
// self-play, candidate-as-Player-One, candidate-as-Player-Two. This preserves
// exact 5,000/2,500/2,500 quotas at the canonical 10,000-game size while also
// defining every small smoke size without special cases.
inline ActorQuotas planned_quotas(int round, std::size_t games) {
  if (round < 0 || round > 2 || games == 0U) {
    throw std::invalid_argument("invalid continuation quota request");
  }
  ActorQuotas quotas{};
  if (round == 0) {
    quotas[static_cast<std::size_t>(ActorMode::Rank4VsRank4)] = games;
    return quotas;
  }

  constexpr std::array<std::size_t, 3> weights{{2U, 1U, 1U}};
  constexpr std::array<ActorMode, 3> modes{{
      ActorMode::CandidateSelfplay,
      ActorMode::CandidatePlayerOne,
      ActorMode::CandidatePlayerTwo,
  }};
  const std::size_t quotient = games / 4U;
  const std::size_t remainder = games % 4U;
  std::size_t assigned = 0U;
  std::array<std::pair<std::size_t, ActorMode>, 3> residuals{};
  for (std::size_t index = 0; index < modes.size(); ++index) {
    const std::size_t count =
        quotient * weights[index] + (remainder * weights[index]) / 4U;
    quotas[static_cast<std::size_t>(modes[index])] = count;
    assigned += count;
    residuals[index] = {(remainder * weights[index]) % 4U, modes[index]};
  }
  std::stable_sort(
      residuals.begin(), residuals.end(),
      [](const auto &left, const auto &right) {
        return left.first > right.first;
      });
  for (std::size_t index = 0; assigned < games; ++index, ++assigned) {
    ++quotas[static_cast<std::size_t>(residuals[index].second)];
  }
  return quotas;
}

}  // namespace papersoccer::jacek_replay_continuations
