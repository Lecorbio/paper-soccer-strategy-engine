#pragma once

#include <vector>

#include "papersoccer/types.hpp"

namespace papersoccer {

bool is_regular_point(const RulesConfig &config, Point point);

bool is_goal_point(const RulesConfig &config, Point point);

bool is_attacking_goal(const RulesConfig &config, Point point, Player player);

bool is_boundary_point(const RulesConfig &config, Point point);

bool is_neighbor(Point from, Point to);

bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment);

std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player);

}  // namespace papersoccer
