#include "papersoccer/debug.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "papersoccer/geometry.hpp"

namespace papersoccer {

namespace {

int min_y(const RulesConfig &) { return -1; }

int max_y(const RulesConfig &config) { return config.height + 1; }

int row_count(const RulesConfig &config) {
  const int points = max_y(config) - min_y(config) + 1;
  return points * 2 - 1;
}

int col_count(const RulesConfig &config) { return (config.width + 1) * 2 - 1; }

int mouth_left_x(const RulesConfig &config) { return (config.width / 2) - 1; }

int mouth_right_x(const RulesConfig &config) { return (config.width / 2) + 1; }

bool is_renderable_point(const RulesConfig &config, Point point) {
  return is_regular_point(config, point) || is_goal_point(config, point);
}

int to_row(const RulesConfig &config, Point point) {
  return (point.y - min_y(config)) * 2;
}

int to_col(Point point) { return point.x * 2; }

char segment_char(Segment segment) {
  const int dx = segment.b.x - segment.a.x;
  const int dy = segment.b.y - segment.a.y;
  if (dx == 0) {
    return '|';
  }
  if (dy == 0) {
    return '-';
  }
  return (dx > 0) == (dy > 0) ? '\\' : '/';
}

std::string status_to_string(Status status) {
  switch (status) {
    case Status::InProgress:
      return "InProgress";
    case Status::WonByOne:
      return "WonByOne";
    case Status::WonByTwo:
      return "WonByTwo";
  }
  return "Unknown";
}

std::string player_to_string(Player player) {
  return player == Player::One ? "One" : "Two";
}

void put(std::vector<std::string> &grid, int row, int col, char value) {
  if (row < 0 || col < 0) {
    return;
  }
  if (row >= static_cast<int>(grid.size())) {
    return;
  }
  if (col >= static_cast<int>(grid.front().size())) {
    return;
  }
  grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = value;
}

}  // namespace

std::string render_ascii(const GameState &state) {
  const int rows = row_count(state.config);
  const int cols = col_count(state.config);
  std::vector<std::string> grid(static_cast<std::size_t>(rows),
                                std::string(static_cast<std::size_t>(cols), ' '));

  for (int x = 0; x < state.config.width; ++x) {
    const bool in_goal_opening = (x >= mouth_left_x(state.config) && x < mouth_right_x(state.config));
    if (!in_goal_opening) {
      put(grid, to_row(state.config, Point{x, 0}), to_col(Point{x, 0}) + 1, '=');
      put(grid, to_row(state.config, Point{x, state.config.height}),
          to_col(Point{x, state.config.height}) + 1, '=');
    }
  }
  for (int y = 0; y < state.config.height; ++y) {
    put(grid, to_row(state.config, Point{0, y}) + 1, to_col(Point{0, y}), '!');
    put(grid, to_row(state.config, Point{state.config.width, y}) + 1,
        to_col(Point{state.config.width, y}), '!');
  }

  for (int x = mouth_left_x(state.config); x < mouth_right_x(state.config); ++x) {
    put(grid, to_row(state.config, Point{x, -1}), to_col(Point{x, -1}) + 1, '=');
    put(grid, to_row(state.config, Point{x, state.config.height + 1}),
        to_col(Point{x, state.config.height + 1}) + 1, '=');
  }
  put(grid, to_row(state.config, Point{mouth_left_x(state.config), -1}) + 1,
      to_col(Point{mouth_left_x(state.config), -1}), '!');
  put(grid, to_row(state.config, Point{mouth_right_x(state.config), -1}) + 1,
      to_col(Point{mouth_right_x(state.config), -1}), '!');
  put(grid, to_row(state.config, Point{mouth_left_x(state.config), state.config.height}) + 1,
      to_col(Point{mouth_left_x(state.config), state.config.height}), '!');
  put(grid, to_row(state.config, Point{mouth_right_x(state.config), state.config.height}) + 1,
      to_col(Point{mouth_right_x(state.config), state.config.height}), '!');

  for (const Segment &segment : state.used_segments) {
    if (!is_renderable_point(state.config, segment.a) ||
        !is_renderable_point(state.config, segment.b)) {
      continue;
    }
    const int row = (to_row(state.config, segment.a) + to_row(state.config, segment.b)) / 2;
    const int col = (to_col(segment.a) + to_col(segment.b)) / 2;
    put(grid, row, col, segment_char(segment));
  }

  for (int y = 0; y <= state.config.height; ++y) {
    for (int x = 0; x <= state.config.width; ++x) {
      const Point point{x, y};
      const char marker = is_boundary_point(state.config, point) ? 'o' : '.';
      put(grid, to_row(state.config, point), to_col(point), marker);
    }
  }

  for (int x = mouth_left_x(state.config); x <= mouth_right_x(state.config); ++x) {
    const Point north_goal{x, -1};
    const Point south_goal{x, state.config.height + 1};
    put(grid, to_row(state.config, north_goal), to_col(north_goal), '^');
    put(grid, to_row(state.config, south_goal), to_col(south_goal), 'v');
  }

  if (is_renderable_point(state.config, state.ball)) {
    put(grid, to_row(state.config, state.ball), to_col(state.ball), '@');
  }

  std::ostringstream out;
  out << "Turn: " << player_to_string(state.to_move) << "\n";
  out << "Status: " << status_to_string(state.status) << "\n";
  out << "Ball: (" << state.ball.x << ", " << state.ball.y << ")\n";
  out << "Legend: @ ball, . interior, o boundary, ^ north goal, v south goal, =/! walls\n";
  out << "\n";

  for (int row = 0; row < rows; ++row) {
    if (row % 2 == 0) {
      const int y = min_y(state.config) + row / 2;
      out << std::setw(3) << y << " ";
    } else {
      out << "    ";
    }
    out << grid[static_cast<std::size_t>(row)] << "\n";
  }
  out << "    ";
  for (int x = 0; x <= state.config.width; ++x) {
    out << x;
    if (x != state.config.width) {
      out << ' ';
    }
  }
  out << "\n";

  return out.str();
}

}  // namespace papersoccer
