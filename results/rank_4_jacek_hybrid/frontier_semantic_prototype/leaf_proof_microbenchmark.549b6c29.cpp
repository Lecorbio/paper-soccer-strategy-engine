#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifndef FRONTIER_IMPL
#error "FRONTIER_IMPL must name the generated submission source"
#endif

#define private public
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include FRONTIER_IMPL
#undef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#undef private

namespace ps = papersoccer;
namespace cg = papersoccer::turn_action_v2;

namespace {

constexpr std::size_t kFixtureCount = 512;

ps::RulesConfig rules() {
  return {8, 10, ps::GoalRule::OwnGoalsAllowed,
          ps::BlockedRule::MoverLoses};
}

std::uint64_t mix(std::uint64_t value) {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

std::vector<std::string_view> split(std::string_view transcript) {
  std::vector<std::string_view> turns;
  std::size_t begin = 0;
  while (begin < transcript.size()) {
    const std::size_t slash = transcript.find('/', begin);
    turns.push_back(transcript.substr(
        begin, slash == std::string_view::npos ? transcript.size() - begin
                                               : slash - begin));
    if (slash == std::string_view::npos) {
      break;
    }
    begin = slash + 1;
  }
  return turns;
}

ps::GameState reconstruct(std::string_view transcript) {
  ps::GameState state = ps::make_initial_state(rules());
  for (const std::string_view action : split(transcript)) {
    cg::apply_encoded_turn(state, action);
  }
  return state;
}

void bind_fixture_digest(std::uint64_t &digest, const ps::GameState &state,
                         std::size_t ordinal) {
  const std::uint64_t packed_ball =
      (static_cast<std::uint64_t>(state.ball.x + 4) << 32U) ^
      static_cast<std::uint64_t>(state.ball.y + 4);
  digest = mix(digest ^ mix(packed_ball) ^
               (static_cast<std::uint64_t>(state.to_move) << 8U) ^
               (static_cast<std::uint64_t>(state.status) << 16U) ^
               (static_cast<std::uint64_t>(state.used_segments.size()) << 24U) ^
               (static_cast<std::uint64_t>(state.visit_count.size()) << 40U) ^
               ordinal);
}

std::vector<ps::GameState> fixtures(std::uint64_t &fixture_digest) {
  constexpr std::array<std::string_view, 8> tactical_transcripts{{
      "",
      "0/6",
      "6/1",
      "0/6/5/4/5/53/61/0633",
      "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35",
      "0/2/7/45/7/5/71/34/2212/2/7/1/6/1/03636074/33535",
      "7/6/7/53/10/34/71/45/221/2/1/35/70/54/17/43/660/33",
      "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/532/27412/41/654",
  }};

  std::vector<ps::GameState> result;
  result.reserve(kFixtureCount);
  fixture_digest = 0x6a09e667f3bcc909ULL;
  for (const std::string_view transcript : tactical_transcripts) {
    ps::GameState state = reconstruct(transcript);
    if (ps::is_terminal(state)) {
      throw std::logic_error("tactical fixture is terminal");
    }
    bind_fixture_digest(fixture_digest, state, result.size());
    result.push_back(std::move(state));
  }

  std::uint64_t random = 0x4f1bbcdc676f2b31ULL;
  while (result.size() < kFixtureCount) {
    ps::GameState state = ps::make_initial_state(rules());
    while (!ps::is_terminal(state) && result.size() < kFixtureCount) {
      bind_fixture_digest(fixture_digest, state, result.size());
      result.push_back(state);
      const std::vector<ps::Move> legal = ps::legal_moves(state);
      if (legal.empty()) {
        throw std::logic_error("live procedural fixture has no legal move");
      }
      state = ps::apply_move(state,
                             legal[next_random(random) % legal.size()]);
    }
  }
  return result;
}

std::uint64_t bind_result(std::uint64_t digest, int score,
                          std::size_t ordinal) {
  const std::uint64_t encoded = static_cast<std::uint64_t>(
      static_cast<std::int64_t>(score) - std::numeric_limits<int>::min());
  return mix(digest ^ mix(encoded) ^ ordinal);
}

}  // namespace

int main() {
  std::uint64_t fixture_digest = 0;
  const std::vector<ps::GameState> corpus = fixtures(fixture_digest);

  cg::SearchConfig config;
  config.max_nodes = 1'000'000;
  config.max_time_ms = 0;
  config.transposition_entries = 4'096;
  config.evaluation_entries = 2'048;
  config.exact_proof_mask = cg::kExactProofLeafBoundary;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;

  std::vector<std::unique_ptr<cg::CompleteTurnSearch>> searches;
  searches.reserve(corpus.size());
  for (const ps::GameState &state : corpus) {
    searches.push_back(std::make_unique<cg::CompleteTurnSearch>(state, config));
  }

  std::uint64_t result_digest = 0xbb67ae8584caa73bULL;
  const auto started = std::chrono::steady_clock::now();
  for (std::size_t index = 0; index < searches.size(); ++index) {
    const int score = searches[index]->search(
        0, 0, -cg::kInfinity, cg::kInfinity);
    result_digest = bind_result(result_digest, score, index);
  }
  const auto stopped = std::chrono::steady_clock::now();

  std::uint64_t probes = 0;
  std::uint64_t wins = 0;
  std::uint64_t losses = 0;
  std::uint64_t evaluation_probes = 0;
  std::uint64_t evaluation_hits = 0;
  for (const auto &search : searches) {
    const cg::SearchStats &stats = search->stats();
    probes += stats.leaf_rebound_probes;
    wins += stats.leaf_rebound_win_hits;
    losses += stats.leaf_rebound_loss_hits;
    evaluation_probes += stats.evaluation_cache_probes;
    evaluation_hits += stats.evaluation_cache_hits;
  }

  if (corpus.size() != kFixtureCount || probes != kFixtureCount ||
      evaluation_probes != kFixtureCount || evaluation_hits != 0 ||
      wins + losses >= kFixtureCount) {
    std::cerr << "leaf-proof fixture coverage invariant failed\n";
    return 2;
  }

  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
      stopped - started).count();
  std::cout << elapsed << ' ' << fixture_digest << ' ' << result_digest << ' '
            << corpus.size() << ' ' << probes << ' ' << wins << ' ' << losses
            << ' ' << evaluation_probes << ' ' << evaluation_hits << '\n';
}
