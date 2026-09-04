#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "submissions/codingame/bots/compact_value_bfm/engine.hpp"

namespace papersoccer::compact_value_bfm_runtime {

inline constexpr const char *kRuntimeSchema =
    "papersoccer.compact-value-bfm-runtime.v1";
inline constexpr const char *kFeatureSchema =
    "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
    "mover-relative-rotate180:true-turn-distance+free-degree";

struct Identity {
  std::string runtime_sha256;
  std::string body_sha256;
  std::string payload_sha256;
  std::string source_bundle_body_sha256;
  std::string selection_sha256;
  std::string architecture;
  std::string arm;
  std::uint64_t seed{};
};

struct LoadedRuntime {
  Identity identity;
  std::unique_ptr<::compact_value_bfm::QuantizedModel> model;
};

// Loads the canonical, body-hashed runtime document and constructs the exact
// signed-three-bit inference model.  No generated model header is involved.
LoadedRuntime load(const std::filesystem::path &path);

}  // namespace papersoccer::compact_value_bfm_runtime
