#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer::opening_bank {

inline constexpr std::string_view kSchema = "papersoccer.opening-bank.v1";
inline constexpr std::string_view kRules =
    "8x10;opponent_goal_only;player_to_move_loses";
inline constexpr std::string_view kGenerator =
    "uniform-legal-move-generator/v1";
inline constexpr std::string_view kSelection =
    "splitmix64-unbiased-rejection-sampling/v1";
inline constexpr std::string_view kStateHashAlgorithm =
    "sha256-canonical-game-state/v1";
inline constexpr std::string_view kCanonicalization =
    "horizontal-reflection-min-serialization-sha256/v1";
inline constexpr std::string_view kOpeningPlyDefinition =
    "one physical selected edge, including rebound edges";

struct OpeningRecord {
  std::string opening_id{};
  std::string phase{};
  std::size_t depth{};
  std::uint64_t generation_seed{};
  std::string state_hash{};
  std::string canonical_key{};
  Player to_move{Player::One};
  std::vector<Move> moves{};

  bool operator==(const OpeningRecord &) const noexcept = default;
};

struct Bank {
  std::string phase{};
  std::size_t depth{};
  std::size_t pairs{};
  std::uint64_t generator_seed{};
  std::vector<OpeningRecord> records{};

  bool operator==(const Bank &) const noexcept = default;
};

bool valid_phase(std::string_view phase) noexcept;
std::string state_hash(const GameState &state);
std::string canonical_key(const GameState &state);
std::string stable_opening_id(std::string_view phase, std::size_t depth,
                              std::string_view hash);
GameState replay_transcript(const std::vector<Move> &moves);

Bank generate_bank(std::string phase, std::size_t depth, std::size_t pairs,
                   std::uint64_t seed,
                   const std::vector<Bank> &excluded_banks = {});
std::string render_bank(const Bank &bank);
Bank parse_bank_text(std::string_view text, std::string_view source_name);
Bank load_bank(const std::filesystem::path &path);
void validate_disjoint(const std::vector<Bank> &banks);

}  // namespace papersoccer::opening_bank
