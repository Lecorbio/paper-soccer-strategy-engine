#pragma once

#include <array>
#include <cstddef>

namespace papersoccer::turn_action_v2::teacher_residual_model {

inline constexpr std::size_t kInputCount = 24;
inline constexpr std::array<float, kInputCount> kWeights{{-0.000212487676F, -0.0314205534F, 0.00000000F, 0.0220964056F, 0.00318384444F, -0.0459056867F, 0.00000000F, 0.00000000F, -0.000812118322F, -0.0889680974F, -0.0757096574F, 0.00821899013F, -0.163250396F, -0.206006910F, 0.0891054922F, 0.00699152156F, -0.000185926722F, 0.0146061040F, 0.127743796F, 0.0246996844F, 0.00000000F, 0.00000000F, 0.00000000F, 0.00474756504F}};
inline constexpr float kBias = -0.0212487676F;

}  // namespace papersoccer::turn_action_v2::teacher_residual_model
