#pragma once

#include <cstddef>
#include <string_view>

namespace compact_value_bfm::model {

// Build-only neutral placeholder.  The strict exporter replaces this header
// byte-for-byte from the selected, body-hashed runtime before qualification.
inline constexpr std::size_t kInputs = 6301;
inline constexpr std::size_t kHiddenOne = 8;
inline constexpr std::size_t kHiddenTwo = 8;
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = 50'480;
inline constexpr std::size_t kPackedByteCount = 0;
inline constexpr float kScaleOne = 1.0F;
inline constexpr float kScaleTwo = 1.0F;
inline constexpr float kScaleThree = 1.0F;
inline constexpr bool kBootstrapZero = true;
inline constexpr std::string_view kRuntimeSchema =
    "papersoccer.compact-value-bfm-runtime.v1";
inline constexpr std::string_view kFeatureSchema =
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree";
inline constexpr std::string_view kPayloadSha256 =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
inline constexpr std::string_view kRuntimeBodySha256 =
    "0000000000000000000000000000000000000000000000000000000000000000";
inline constexpr std::string_view kIdentity = "bootstrap-zero-not-qualified";
inline constexpr std::string_view kPackedWeights = "";

}  // namespace compact_value_bfm::model
