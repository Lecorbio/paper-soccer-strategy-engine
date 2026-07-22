#pragma once

#include <array>
#include <cstddef>

namespace papersoccer::turn_action_v2::teacher_residual_model {

inline constexpr std::size_t kInputCount = 24;
inline constexpr std::array<float, kInputCount> kWeights{{0.0000438559672F, -0.0482640395F, 0.00000000F, 0.00965929746F, 0.00487774885F, -0.0376609406F, 0.00000000F, 0.00000000F, -0.00658001635F, -0.0542466760F, -0.0465506819F, 0.00720994156F, -0.191482948F, -0.374691793F, 0.264679235F, -0.131914693F, 0.0000383721406F, 0.0112093268F, 0.0894050913F, 0.0175712674F, 0.00000000F, 0.00000000F, 0.00000000F, -0.00437146101F}};
inline constexpr float kBias = 0.00438559637F;

}  // namespace papersoccer::turn_action_v2::teacher_residual_model
