#pragma once

#include <cstdint>
#include <string_view>

#include "papersoccer/types.hpp"

namespace papersoccer {

class Bot {
 public:
  virtual ~Bot() = default;

  virtual std::string_view name() const noexcept = 0;
  virtual Move choose_move(const GameState &state) = 0;
};

class RandomBot final : public Bot {
 public:
  static constexpr std::uint64_t default_seed() noexcept { return 0xC0FFEE1234ULL; }

  explicit RandomBot(std::uint64_t seed = default_seed()) noexcept;

  std::string_view name() const noexcept override;
  Move choose_move(const GameState &state) override;

 private:
  std::uint64_t state_;

  std::uint64_t next_random() noexcept;
};

}  // namespace papersoccer
