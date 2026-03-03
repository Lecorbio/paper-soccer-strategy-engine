#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace papersoccer {

enum class Player { One, Two };

enum class Status { InProgress, WonByOne, WonByTwo };

inline constexpr Player opponent(Player player) noexcept {
  return player == Player::One ? Player::Two : Player::One;
}

struct Point {
  int x{0};
  int y{0};

  constexpr bool operator==(const Point &) const noexcept = default;
};

inline constexpr bool operator<(const Point &lhs, const Point &rhs) noexcept {
  return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
}

struct PointHash {
  std::size_t operator()(const Point &point) const noexcept {
    const auto x = static_cast<std::uint64_t>(static_cast<std::uint32_t>(point.x));
    const auto y = static_cast<std::uint64_t>(static_cast<std::uint32_t>(point.y));
    return static_cast<std::size_t>((x << 32U) ^ y);
  }
};

struct Segment {
  Point a{};
  Point b{};

  Segment() = default;

  Segment(Point first, Point second) noexcept {
    if (second < first) {
      std::swap(first, second);
    }
    a = first;
    b = second;
  }

  constexpr bool operator==(const Segment &) const noexcept = default;
};

struct SegmentHash {
  std::size_t operator()(const Segment &segment) const noexcept {
    PointHash hash_point;
    const auto first = hash_point(segment.a);
    const auto second = hash_point(segment.b);
    return first ^ (second + 0x9e3779b97f4a7c15ULL + (first << 6U) + (first >> 2U));
  }
};

struct Move {
  Point to{};
  constexpr bool operator==(const Move &) const noexcept = default;
};

struct RulesConfig {
  int width{8};
  int height{10};
};

struct GameState {
  RulesConfig config{};
  Point ball{};
  Player to_move{Player::One};
  Status status{Status::InProgress};
  std::vector<Point> path{};
  std::unordered_set<Segment, SegmentHash> used_segments{};
  std::unordered_map<Point, int, PointHash> visit_count{};
};

}  // namespace papersoccer
