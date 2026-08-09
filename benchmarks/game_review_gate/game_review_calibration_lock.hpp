#pragma once

#include <cstdint>

namespace papersoccer::game_review_lock {

struct Calibration {
  const char *identity;
  const char *evidence_profile_id;
  const char *search_profile_name;
  const char *profile_sha256;
  const char *mapping_sha256;
  double raw_intercept;
  double raw_score_coefficient;
};

inline constexpr std::uint64_t selected_deep_nodes = 400000ULL;

inline constexpr Calibration fast_calibration{
    "complete-turn-analysis-fast-50k-validation-logistic-v1",
    "complete-turn-analysis-fast-50k",
    "fast-50k",
    "1154ddb51388b2597c459a58a58ec7e2ebf1a2e7fc18d2f4a69df04662d4cc65",
    "ba72f86d1cfea3d670d85681131d25ff9a0eb79b82334da894d82bf3f7420bc5",
    -0.39627788039748957,
    0.00030568558814047836,
};

inline constexpr Calibration deep_calibration{
    "deep-turn-search-400k-validation-logistic-v1",
    "deep-turn-search-400k",
    "deep-400k",
    "73f66e301ff143ecfe5a4d136e9d432a9ba1ef0836628b0fe55faab4c50c770e",
    "0ca9c51bf92b2848c858a8d0c4ff7c7104bfb55477b95e997cb232547592e9d4",
    0.61366459368045656,
    4.2319442291347914e-05,
};

}  // namespace papersoccer::game_review_lock
