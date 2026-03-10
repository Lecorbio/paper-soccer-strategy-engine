#include "papersoccer/geometry.hpp"

#include <algorithm>
#include <cmath>

namespace papersoccer {

namespace {

int center_x(const RulesConfig &config) { return config.width / 2; }

int mouth_left_x(const RulesConfig &config) { return center_x(config) - 1; }

int mouth_right_x(const RulesConfig &config) { return center_x(config) + 1; }

bool is_mouth_x(const RulesConfig &config, int x) {
  return x >= mouth_left_x(config) && x <= mouth_right_x(config);
}

bool is_north_goal(const RulesConfig &config, Point point) {
  return is_mouth_x(config, point.x) && point.y == -1;
}

bool is_south_goal(const RulesConfig &config, Point point) {
  return is_mouth_x(config, point.x) && point.y == config.height + 1;
}

bool is_goal_mouth_point(const RulesConfig &config, Point point) {
  return is_mouth_x(config, point.x) && (point.y == 0 || point.y == config.height);
}

bool is_north_goal_post_segment(const RulesConfig &config, Segment segment) {
  const bool touches_north_goal = is_north_goal(config, segment.a) || is_north_goal(config, segment.b);
  if (!touches_north_goal) {
    return false;
  }

  return segment.a.x == segment.b.x &&
         (segment.a.x == mouth_left_x(config) || segment.a.x == mouth_right_x(config)) &&
         ((segment.a.y == -1 && segment.b.y == 0) || (segment.a.y == 0 && segment.b.y == -1));
}

bool is_south_goal_post_segment(const RulesConfig &config, Segment segment) {
  const bool touches_south_goal = is_south_goal(config, segment.a) || is_south_goal(config, segment.b);
  if (!touches_south_goal) {
    return false;
  }

  return segment.a.x == segment.b.x &&
         (segment.a.x == mouth_left_x(config) || segment.a.x == mouth_right_x(config)) &&
         ((segment.a.y == config.height && segment.b.y == config.height + 1) ||
          (segment.a.y == config.height + 1 && segment.b.y == config.height));
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
  if (is_north_goal_post_segment(config, segment) || is_south_goal_post_segment(config, segment)) {
    return true;
  }

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
      for (int goal_x = mouth_left_x(config); goal_x <= mouth_right_x(config); ++goal_x) {
        const Point goal_point{goal_x, -1};
        if (is_neighbor(from, goal_point)) {
          result.push_back(goal_point);
        }
      }
    }
    if (player == Player::Two && from.y == config.height) {
      for (int goal_x = mouth_left_x(config); goal_x <= mouth_right_x(config); ++goal_x) {
        const Point goal_point{goal_x, config.height + 1};
        if (is_neighbor(from, goal_point)) {
          result.push_back(goal_point);
        }
      }
    }
  }

  return result;
}

}  // namespace papersoccer
