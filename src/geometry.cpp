#include "papersoccer/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace papersoccer {

namespace {

int center_x(const RulesConfig &config) { return config.width / 2; }

bool is_north_goal(const RulesConfig &config, Point point) {
  return point.x == center_x(config) && point.y == -1;
}

bool is_south_goal(const RulesConfig &config, Point point) {
  return point.x == center_x(config) && point.y == config.height + 1;
}

bool is_goal_mouth_point(const RulesConfig &config, Point point) {
  const int mouth_left = center_x(config) - 1;
  const int mouth_right = center_x(config) + 1;
  const bool in_mouth_x = point.x >= mouth_left && point.x <= mouth_right;
  return in_mouth_x && (point.y == 0 || point.y == config.height);
}

}  // namespace

bool is_regular_point(const RulesConfig &config, Point point) {
  return point.x >= 0 && point.x <= config.width && point.y >= 0 &&
         point.y <= config.height;
}

bool is_goal_point(const RulesConfig &config, Point point) {
  return is_north_goal(config, point) || is_south_goal(config, point);
}

bool is_boundary_point(const RulesConfig &config, Point point) {
  if (!is_regular_point(config, point)) {
    return false;
  }
  return point.x == 0 || point.x == config.width || point.y == 0 ||
         point.y == config.height;
}

bool is_neighbor(Point from, Point to) {
  const int dx = std::abs(from.x - to.x);
  const int dy = std::abs(from.y - to.y);
  return dx <= 1 && dy <= 1 && (dx + dy) > 0;
}

bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment) {
  if (!is_regular_point(config, segment.a) || !is_regular_point(config, segment.b)) {
    return false;
  }

  if (!(is_boundary_point(config, segment.a) && is_boundary_point(config, segment.b))) {
    return false;
  }

  const int dx = std::abs(segment.a.x - segment.b.x);
  const int dy = std::abs(segment.a.y - segment.b.y);

  if (segment.a.y == segment.b.y && (segment.a.y == 0 || segment.a.y == config.height) &&
      dx == 1 && dy == 0) {
    return true;
  }

  if (segment.a.x == segment.b.x && (segment.a.x == 0 || segment.a.x == config.width) &&
      dx == 0 && dy == 1) {
    return true;
  }

  return false;
}

std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player) {
  std::vector<Point> result;
  if (!is_regular_point(config, from)) {
    return result;
  }

  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const Point candidate{from.x + dx, from.y + dy};
      if (is_regular_point(config, candidate)) {
        result.push_back(candidate);
      }
    }
  }

  if (is_goal_mouth_point(config, from)) {
    if (player == Player::One && from.y == 0) {
      result.push_back(Point{center_x(config), -1});
    }
    if (player == Player::Two && from.y == config.height) {
      result.push_back(Point{center_x(config), config.height + 1});
    }
  }

  return result;
}

}  // namespace papersoccer
