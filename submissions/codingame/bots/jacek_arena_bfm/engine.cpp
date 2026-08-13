#include "engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "model.hpp"

namespace jacek_arena_bfm {
namespace {

static_assert(model::kInputSize == kFeatureCount,
              "model input must match mover-relative features");
static_assert(model::kHidden1Size == 32 || model::kHidden1Size == 48 ||
                  model::kHidden1Size == 64,
              "unsupported first hidden-layer width");
static_assert(model::kHidden2Size == 32,
              "second hidden layer must have width 32");

constexpr std::array<int, 8> kDx{0, 1, 1, 1, 0, -1, -1, -1};
constexpr std::array<int, 8> kDy{-1, -1, 0, 1, 1, 1, 0, -1};

int direction_for(int dx, int dy) {
  for (int direction = 0; direction < 8; ++direction) {
    if (kDx[direction] == dx && kDy[direction] == dy) {
      return direction;
    }
  }
  return -1;
}

bool regular(int x, int y) {
  return x >= 0 && x <= 8 && y >= 1 && y <= 11;
}

bool goal(int x, int y) {
  return x >= 3 && x <= 5 && (y == 0 || y == 12);
}

bool boundary_coordinate(int x, int y) {
  if (!regular(x, y)) {
    return false;
  }
  return x == 0 || x == 8 ||
         ((y == 1 || y == 11) && x != 4);
}

bool permitted_edge(int ax, int ay, int bx, int by) {
  const int dx = std::abs(ax - bx);
  const int dy = std::abs(ay - by);
  if (std::max(dx, dy) != 1) {
    return false;
  }
  if (regular(ax, ay) && regular(bx, by)) {
    if (boundary_coordinate(ax, ay) && boundary_coordinate(bx, by)) {
      if (ay == by && (ay == 1 || ay == 11)) {
        return false;
      }
      if (ax == bx && (ax == 0 || ax == 8)) {
        return false;
      }
    }
    return true;
  }
  // A goal is terminal.  Its back and side walls are not playable edges, and
  // paths cannot cross a post from outside the three-point mouth.
  if (goal(ax, ay) == goal(bx, by)) {
    return false;
  }
  const int field_x = regular(ax, ay) ? ax : bx;
  const int field_y = regular(ax, ay) ? ay : by;
  const int goal_x = goal(ax, ay) ? ax : bx;
  const int goal_y = goal(ax, ay) ? ay : by;
  if (field_x < 3 || field_x > 5 ||
      !((field_y == 1 && goal_y == 0) ||
        (field_y == 11 && goal_y == 12))) {
    return false;
  }
  return !(field_x == goal_x && (field_x == 3 || field_x == 5));
}

std::uint64_t mix64(std::uint64_t value) noexcept {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

bool same_state(const State &left, const State &right) noexcept {
  return left.used == right.used && left.ball == right.ball &&
         left.to_move == right.to_move && left.winner == right.winner;
}

std::size_t action_limit(GeneratorStrategy strategy, bool root) {
  switch (strategy) {
    case GeneratorStrategy::Fixed250NineOne:
      return 250;
    case GeneratorStrategy::TacticalProgressive:
      return root ? 512 : 256;
    case GeneratorStrategy::PriorityBeam:
      return root ? 512 : 256;
    case GeneratorStrategy::HighCapRecall:
      return 8192;
  }
  return 256;
}

int mobility(const State &state) {
  if (state.terminal()) {
    return 0;
  }
  int result = 0;
  const auto &topology = Topology::get();
  for (int index = 0; index < topology.degree(state.ball); ++index) {
    if (!edge_used(state, topology.arc(state.ball, index).edge)) {
      ++result;
    }
  }
  return result;
}

int tactical_score(const State &state, const Action &action,
                   std::uint8_t mover) {
  if (state.terminal()) {
    return state.winner == static_cast<int>(mover) ? 1000000 : -1000000;
  }
  const auto &topology = Topology::get();
  const int relative_y = mover == 0 ? topology.y(state.ball)
                                    : 12 - topology.y(state.ball);
  int score = (12 - relative_y) * 64 - mobility(state) * 3;
  score -= static_cast<int>(action.length);
  if (topology.boundary(state.ball)) {
    score += 40;
  }
  return score;
}

struct Partial {
  State state{};
  Action action{};
  int priority{0};
  std::uint64_t serial{0};
};

struct HigherPriority {
  bool operator()(const Partial &left, const Partial &right) const noexcept {
    if (left.priority != right.priority) {
      return left.priority < right.priority;
    }
    return left.serial > right.serial;
  }
};

bool append_direction(Action &action, std::uint8_t direction) {
  if (action.length >= kMaximumActionLength) {
    return false;
  }
  action.directions[action.length++] = direction;
  return true;
}

bool terminal_for_mover(const State &state, std::uint8_t mover) {
  return state.terminal() && state.winner == static_cast<int>(mover);
}

std::uint8_t mover_relative_direction(std::uint8_t direction,
                                      std::uint8_t mover) noexcept {
  return mover == 0 ? direction
                    : static_cast<std::uint8_t>((direction + 4U) & 7U);
}

int mover_relative_edge(int edge, std::uint8_t mover) noexcept {
  return mover == 0 ? edge : Topology::get().rotated_edge(edge);
}

std::uint64_t mover_relative_hash(const State &state, std::uint8_t mover) {
  return state_hash(mover == 0 ? state : rotate_and_swap(state));
}

std::string mover_relative_action_key(const Action &action,
                                      std::uint8_t mover) {
  std::string result;
  result.reserve(action.length);
  for (std::size_t index = 0; index < action.length; ++index) {
    result.push_back(static_cast<char>(
        '0' + mover_relative_direction(action.directions[index], mover)));
  }
  return result;
}

std::vector<std::int8_t> decode_base64(const char *text,
                                        std::size_t expected) {
  std::vector<std::int8_t> result;
  result.reserve(expected);
  std::uint32_t accumulator = 0;
  int bits = 0;
  for (const unsigned char character : std::string(text)) {
    int value = -1;
    if (character >= 'A' && character <= 'Z') value = character - 'A';
    if (character >= 'a' && character <= 'z') value = character - 'a' + 26;
    if (character >= '0' && character <= '9') value = character - '0' + 52;
    if (character == '+') value = 62;
    if (character == '/') value = 63;
    if (value < 0) {
      if (character == '=') break;
      continue;
    }
    accumulator = (accumulator << 6U) | static_cast<std::uint32_t>(value);
    bits += 6;
    while (bits >= 8) {
      bits -= 8;
      const auto byte = static_cast<unsigned char>((accumulator >> bits) & 255U);
      result.push_back(static_cast<std::int8_t>(byte));
    }
    accumulator &= bits == 0 ? 0U : (1U << bits) - 1U;
  }
  if (result.size() != expected) {
    result.clear();
  }
  return result;
}

float bootstrap_weight(std::uint64_t layer, std::size_t index, float scale) {
  const std::uint64_t bits = mix64(model::kBootstrapSeed ^
                                   (layer << 56U) ^ index);
  const float unit = static_cast<float>((bits >> 40U) & 0xffffffU) /
                     8388607.5F - 1.0F;
  return unit * scale;
}

class Network {
 public:
  Network()
      : w1_(decode_base64(model::kW1Packed, model::kW1Count)),
        w2_(decode_base64(model::kW2Packed, model::kW2Count)),
        w3_(decode_base64(model::kW3Packed, model::kW3Count)) {}

  float run(const State &state) const {
    std::size_t feature_count = 0;
    const auto features = active_features(state, feature_count);
    std::array<float, 64> first{};
    std::array<float, 32> second{};
    const bool packed1 = w1_.size() ==
        static_cast<std::size_t>(model::kHidden1Size * model::kInputSize);
    const bool packed2 = w2_.size() ==
        static_cast<std::size_t>(model::kHidden2Size * model::kHidden1Size);
    const bool packed3 = w3_.size() ==
        static_cast<std::size_t>(model::kHidden2Size);
    for (int output = 0; output < model::kHidden1Size; ++output) {
      float sum = 0.0F;
      const std::size_t base = static_cast<std::size_t>(output) *
                               model::kInputSize;
      for (std::size_t item = 0; item < feature_count; ++item) {
        const std::size_t index = base + features[item];
        sum += packed1 ? static_cast<float>(w1_[index]) * model::kW1Scale
                       : bootstrap_weight(1, index, 0.035F);
      }
      first[output] = std::tanh(sum);
    }
    for (int output = 0; output < model::kHidden2Size; ++output) {
      float sum = 0.0F;
      const std::size_t base = static_cast<std::size_t>(output) *
                               model::kHidden1Size;
      for (int input = 0; input < model::kHidden1Size; ++input) {
        const std::size_t index = base + input;
        sum += first[input] *
               (packed2 ? static_cast<float>(w2_[index]) * model::kW2Scale
                        : bootstrap_weight(2, index, 0.16F));
      }
      second[output] = std::tanh(sum);
    }
    float result = 0.0F;
    for (int input = 0; input < model::kHidden2Size; ++input) {
      result += second[input] *
          (packed3 ? static_cast<float>(w3_[input]) * model::kW3Scale
                   : bootstrap_weight(3, input, 0.20F));
    }
    return std::tanh(result);
  }

 private:
  std::vector<std::int8_t> w1_;
  std::vector<std::int8_t> w2_;
  std::vector<std::int8_t> w3_;
};

const Network &network() {
  static const Network instance;
  return instance;
}

float root_value(const State &state, std::uint8_t root_player) {
  if (state.terminal()) {
    return state.winner == static_cast<int>(root_player) ? 1.0F : -1.0F;
  }
  const float value_for_mover = evaluate(state);
  return state.to_move == root_player ? value_for_mover : -value_for_mover;
}

struct TreeNode {
  State state{};
  int parent{-1};
  int action_from_parent{-1};
  std::vector<Action> actions{};
  std::vector<int> children{};
  std::size_t expanded{0};
  std::uint32_t visits{0};
  double total{0.0};
  bool generated{false};
};

}  // namespace

bool Action::operator==(const Action &other) const noexcept {
  if (length != other.length) {
    return false;
  }
  return std::equal(directions.begin(), directions.begin() + length,
                    other.directions.begin());
}

std::string Action::text() const {
  std::string result;
  result.reserve(length);
  for (std::size_t index = 0; index < length; ++index) {
    result.push_back(static_cast<char>('0' + directions[index]));
  }
  return result;
}

const Topology &Topology::get() {
  static const Topology topology;
  return topology;
}

Topology::Topology() {
  int vertex_count = 0;
  for (int x = 3; x <= 5; ++x) {
    xs_[vertex_count] = static_cast<std::int8_t>(x);
    ys_[vertex_count++] = 0;
  }
  for (int y = 1; y <= 11; ++y) {
    for (int x = 0; x <= 8; ++x) {
      xs_[vertex_count] = static_cast<std::int8_t>(x);
      ys_[vertex_count++] = static_cast<std::int8_t>(y);
    }
  }
  for (int x = 3; x <= 5; ++x) {
    xs_[vertex_count] = static_cast<std::int8_t>(x);
    ys_[vertex_count++] = 12;
  }
  if (vertex_count != kVertexCount) {
    throw std::logic_error("jacek_arena_bfm vertex topology mismatch");
  }
  for (int vertex = 0; vertex < kVertexCount; ++vertex) {
    boundaries_[vertex] = boundary_coordinate(xs_[vertex], ys_[vertex]);
  }
  int edge_count = 0;
  std::array<std::array<int, kVertexCount>, kVertexCount> edge_ids{};
  for (auto &row : edge_ids) row.fill(-1);
  for (int first = 0; first < kVertexCount; ++first) {
    for (int second = first + 1; second < kVertexCount; ++second) {
      if (!permitted_edge(xs_[first], ys_[first], xs_[second], ys_[second])) {
        continue;
      }
      if (edge_count >= kEdgeCount) {
        throw std::logic_error("jacek_arena_bfm has too many edges");
      }
      edge_ids[first][second] = edge_ids[second][first] = edge_count++;
    }
  }
  if (edge_count != kEdgeCount) {
    throw std::logic_error("jacek_arena_bfm edge topology mismatch");
  }
  for (int first = 0; first < kVertexCount; ++first) {
    for (int second = 0; second < kVertexCount; ++second) {
      const int edge = edge_ids[first][second];
      if (edge < 0) continue;
      const int direction = direction_for(xs_[second] - xs_[first],
                                          ys_[second] - ys_[first]);
      if (direction < 0 || degrees_[first] >= 8) {
        throw std::logic_error("jacek_arena_bfm invalid adjacency");
      }
      arcs_[first][degrees_[first]++] = Arc{
          static_cast<std::uint16_t>(edge), static_cast<std::uint8_t>(second),
          static_cast<std::uint8_t>(direction)};
    }
    std::sort(arcs_[first].begin(), arcs_[first].begin() + degrees_[first],
              [](const Arc &left, const Arc &right) {
                return left.direction < right.direction;
              });
  }
  for (int vertex = 0; vertex < kVertexCount; ++vertex) {
    const int rotated = vertex_at(8 - xs_[vertex], 12 - ys_[vertex]);
    if (rotated < 0) throw std::logic_error("missing rotated vertex");
    rotated_vertices_[vertex] = static_cast<std::uint8_t>(rotated);
  }
  for (int first = 0; first < kVertexCount; ++first) {
    for (int index = 0; index < degrees_[first]; ++index) {
      const Arc current = arcs_[first][index];
      if (first > current.destination) continue;
      const int rotated_first = rotated_vertices_[first];
      const int rotated_second = rotated_vertices_[current.destination];
      const int rotated = edge_ids[rotated_first][rotated_second];
      if (rotated < 0) throw std::logic_error("missing rotated edge");
      rotated_edges_[current.edge] = static_cast<std::uint16_t>(rotated);
    }
  }
}

int Topology::vertex_at(int x_value, int y_value) const noexcept {
  if (y_value == 0 && x_value >= 3 && x_value <= 5) return x_value - 3;
  if (regular(x_value, y_value)) return 3 + (y_value - 1) * 9 + x_value;
  if (y_value == 12 && x_value >= 3 && x_value <= 5) return 102 + x_value - 3;
  return -1;
}

bool Topology::north_goal(int vertex) const noexcept { return ys_[vertex] == 0; }
bool Topology::south_goal(int vertex) const noexcept { return ys_[vertex] == 12; }

State initial_state() {
  State state;
  state.ball = static_cast<std::uint8_t>(Topology::get().vertex_at(4, 6));
  return state;
}

bool edge_used(const State &state, int edge) noexcept {
  return (state.used[edge >> 6] & (1ULL << (edge & 63))) != 0;
}

bool vertex_visited(const State &state, int vertex) noexcept {
  const auto &topology = Topology::get();
  if (topology.boundary(vertex)) return true;
  for (int index = 0; index < topology.degree(vertex); ++index) {
    if (edge_used(state, topology.arc(vertex, index).edge)) return true;
  }
  return false;
}

std::vector<Topology::Arc> legal_arcs(const State &state) {
  std::vector<Topology::Arc> result;
  if (state.terminal()) return result;
  const auto &topology = Topology::get();
  result.reserve(topology.degree(state.ball));
  for (int index = 0; index < topology.degree(state.ball); ++index) {
    const auto arc = topology.arc(state.ball, index);
    if (!edge_used(state, arc.edge)) result.push_back(arc);
  }
  return result;
}

bool apply_edge(State &state, std::uint8_t direction) {
  if (state.terminal() || direction > 7) return false;
  const auto &topology = Topology::get();
  Topology::Arc selected{};
  bool found = false;
  for (int index = 0; index < topology.degree(state.ball); ++index) {
    const auto arc = topology.arc(state.ball, index);
    if (arc.direction == direction && !edge_used(state, arc.edge)) {
      selected = arc;
      found = true;
      break;
    }
  }
  if (!found) return false;
  const std::uint8_t mover = state.to_move;
  const bool rebound = vertex_visited(state, selected.destination);
  state.used[selected.edge >> 6] |= 1ULL << (selected.edge & 63);
  state.ball = selected.destination;
  ++state.ply;
  if (topology.north_goal(state.ball) || topology.south_goal(state.ball)) {
    const bool attacking = mover == 0 ? topology.north_goal(state.ball)
                                      : topology.south_goal(state.ball);
    state.winner = static_cast<std::int8_t>(attacking ? mover : 1 - mover);
    return true;
  }
  if (!rebound) state.to_move = static_cast<std::uint8_t>(1 - mover);
  if (legal_arcs(state).empty()) {
    state.winner = static_cast<std::int8_t>(1 - mover);
  }
  return true;
}

bool apply_action(State &state, const Action &action, bool require_complete) {
  if (action.length == 0 || action.length > kMaximumActionLength ||
      state.terminal()) return false;
  const std::uint8_t mover = state.to_move;
  for (std::size_t index = 0; index < action.length; ++index) {
    if (state.terminal() || state.to_move != mover) return false;
    if (!apply_edge(state, action.directions[index])) return false;
    if (index + 1 < action.length &&
        (state.terminal() || state.to_move != mover)) return false;
  }
  return !require_complete || state.terminal() || state.to_move != mover;
}

std::uint64_t state_hash(const State &state) noexcept {
  std::uint64_t hash = mix64(state.ball | (static_cast<std::uint64_t>(state.to_move) << 8U) |
                             (static_cast<std::uint64_t>(state.winner + 1) << 16U));
  for (std::size_t index = 0; index < state.used.size(); ++index) {
    hash ^= mix64(state.used[index] + index * 0x9e3779b97f4a7c15ULL);
  }
  return hash;
}

Action emergency_complete_action(const State &source) {
  Action action;
  if (source.terminal()) return action;
  State state = source;
  const std::uint8_t mover = state.to_move;
  while (!state.terminal() && state.to_move == mover &&
         action.length < kMaximumActionLength) {
    const auto arcs = legal_arcs(state);
    if (arcs.empty()) break;
    bool found = false;
    int best_score = std::numeric_limits<int>::min();
    std::uint8_t best_direction = 0;
    for (const auto arc : arcs) {
      State successor = state;
      if (!apply_edge(successor, arc.direction)) continue;
      Action candidate = action;
      if (!append_direction(candidate, arc.direction)) continue;
      const int score = tactical_score(successor, candidate, mover);
      if (!found || score > best_score ||
          (score == best_score &&
           mover_relative_direction(arc.direction, mover) <
               mover_relative_direction(best_direction, mover))) {
        found = true;
        best_score = score;
        best_direction = arc.direction;
      }
    }
    if (!found || !append_direction(action, best_direction) ||
        !apply_edge(state, best_direction)) {
      return {};
    }
  }
  if (action.length == 0 ||
      (!state.terminal() && state.to_move == mover)) {
    return {};
  }
  return action;
}

std::vector<Action> generate_actions(const State &state,
                                     GeneratorStrategy strategy, bool root,
                                     GeneratorStats *stats,
                                     std::chrono::steady_clock::time_point deadline) {
  GeneratorStats local;
  const bool timed = deadline != std::chrono::steady_clock::time_point::max();
  const std::size_t limit = action_limit(strategy, root);
  const std::size_t partial_limit = limit *
      (strategy == GeneratorStrategy::HighCapRecall ? 64 : 32);
  std::vector<Action> completed;
  completed.reserve(limit);
  std::vector<std::pair<std::uint64_t, State>> boundaries;
  boundaries.reserve(limit);
  if (state.terminal()) {
    if (stats) *stats = local;
    return completed;
  }
  const Action emergency = timed ? emergency_complete_action(state) : Action{};
  auto out_of_time = [&]() {
    if (!timed || std::chrono::steady_clock::now() < deadline) return false;
    local.deadline_reached = true;
    return true;
  };
  const std::uint8_t mover = state.to_move;
  std::uint64_t serial = 0;
  std::deque<Partial> deque;
  std::priority_queue<Partial, std::vector<Partial>, HigherPriority> priority;
  Partial start{state, {}, 0, serial++};
  if (strategy == GeneratorStrategy::Fixed250NineOne) deque.push_back(start);
  else priority.push(start);
  std::size_t selections = 0;
  std::array<std::size_t, 8> boundary_classes{};
  auto has_partial = [&] {
    return strategy == GeneratorStrategy::Fixed250NineOne
               ? !deque.empty() : !priority.empty();
  };
  auto pop_partial = [&] {
    Partial value;
    if (strategy == GeneratorStrategy::Fixed250NineOne) {
      ++selections;
      if (selections % 10 == 0) {
        value = deque.front();
        deque.pop_front();
      } else {
        value = deque.back();
        deque.pop_back();
      }
    } else {
      value = priority.top();
      priority.pop();
    }
    return value;
  };
  while (has_partial() && completed.size() < limit &&
         local.partials < partial_limit && !out_of_time()) {
    Partial current = pop_partial();
    ++local.partials;
    auto arcs = legal_arcs(current.state);
    const std::uint64_t order_seed = mover_relative_hash(current.state, mover) ^
                                     (current.action.length * 0x9e3779b9U);
    std::stable_sort(arcs.begin(), arcs.end(), [&](const auto &left,
                                                   const auto &right) {
      State a = current.state;
      State b = current.state;
      apply_edge(a, left.direction);
      apply_edge(b, right.direction);
      int sa = tactical_score(a, current.action, mover);
      int sb = tactical_score(b, current.action, mover);
      if (sa != sb) return sa > sb;
      return mix64(order_seed ^ mover_relative_edge(left.edge, mover)) <
             mix64(order_seed ^ mover_relative_edge(right.edge, mover));
    });
    for (const auto arc : arcs) {
      if ((serial & 7U) == 0U && out_of_time()) break;
      Partial next = current;
      next.serial = serial++;
      if (!append_direction(next.action, arc.direction) ||
          !apply_edge(next.state, arc.direction)) continue;
      if (next.state.terminal() || next.state.to_move != mover) {
        const std::uint64_t hash = state_hash(next.state);
        bool duplicate = false;
        for (const auto &[other_hash, other_state] : boundaries) {
          if (hash == other_hash && same_state(next.state, other_state)) {
            duplicate = true;
            break;
          }
        }
        if (duplicate) {
          ++local.duplicates;
          continue;
        }
        boundaries.push_back({hash, next.state});
        next.priority = tactical_score(next.state, next.action, mover);
        if (strategy == GeneratorStrategy::PriorityBeam &&
            !next.state.terminal()) {
          const auto &topology = Topology::get();
          const int relative_x = mover == 0 ? topology.x(next.state.ball)
                                            : 8 - topology.x(next.state.ball);
          const int relative_y = mover == 0 ? topology.y(next.state.ball)
                                            : 12 - topology.y(next.state.ball);
          const int klass = (relative_x / 3) + 3 * (relative_y / 5);
          const std::size_t safe_class = static_cast<std::size_t>(
              std::clamp(klass, 0, 7));
          next.priority -= static_cast<int>(boundary_classes[safe_class] * 8);
          ++boundary_classes[safe_class];
        }
        completed.push_back(next.action);
        if (completed.size() == limit) break;
      } else {
        next.priority = tactical_score(next.state, next.action, mover) * 4 -
                        next.action.length;
        if (strategy == GeneratorStrategy::Fixed250NineOne) {
          deque.push_back(std::move(next));
        } else {
          priority.push(std::move(next));
        }
      }
    }
  }
  if (completed.empty() && emergency.length != 0) {
    completed.push_back(emergency);
  }
  local.completed = completed.size();
  local.truncated = has_partial() || local.partials >= partial_limit ||
                    local.deadline_reached;
  // Tactical terminals are never displaced by queue order or beam diversity.
  const bool have_sort_reserve = !timed ||
      std::chrono::steady_clock::now() + std::chrono::milliseconds(2) < deadline;
  if (have_sort_reserve) {
    std::stable_sort(completed.begin(), completed.end(), [&](const Action &left,
                                                             const Action &right) {
      State a = state;
      State b = state;
      apply_action(a, left);
      apply_action(b, right);
      const int sa = tactical_score(a, left, mover);
      const int sb = tactical_score(b, right, mover);
      return sa != sb ? sa > sb
                      : mover_relative_action_key(left, mover) <
                            mover_relative_action_key(right, mover);
    });
  } else {
    local.deadline_reached = true;
    local.truncated = true;
  }
  if (stats) *stats = local;
  return completed;
}

std::array<std::uint16_t, 421> active_features(const State &state,
                                                std::size_t &count) {
  std::array<std::uint16_t, 421> result{};
  count = 0;
  const auto &topology = Topology::get();
  for (int edge = 0; edge < kEdgeCount; ++edge) {
    if (edge_used(state, edge)) {
      const int relative_edge = state.to_move == 0 ? edge
                                                   : topology.rotated_edge(edge);
      result[count++] = static_cast<std::uint16_t>(relative_edge);
    }
  }
  constexpr int kInfinity = 10000;
  std::array<int, kVertexCount> distance{};
  distance.fill(kInfinity);
  std::deque<int> queue;
  distance[state.ball] = 0;
  queue.push_front(state.ball);
  while (!queue.empty()) {
    const int source = queue.front();
    queue.pop_front();
    for (int index = 0; index < topology.degree(source); ++index) {
      const auto arc = topology.arc(source, index);
      if (edge_used(state, arc.edge)) continue;
      const int cost = vertex_visited(state, arc.destination) ? 0 : 1;
      if (distance[source] + cost >= distance[arc.destination]) continue;
      distance[arc.destination] = distance[source] + cost;
      if (cost == 0) queue.push_front(arc.destination);
      else queue.push_back(arc.destination);
    }
  }
  for (int absolute_vertex = 0; absolute_vertex < kVertexCount;
       ++absolute_vertex) {
    const int relative_vertex = state.to_move == 0
        ? absolute_vertex : topology.rotated_vertex(absolute_vertex);
    const int bucket = std::min(distance[absolute_vertex], 7);
    result[count++] = static_cast<std::uint16_t>(
        kEdgeCount + relative_vertex * 8 + bucket);
  }
  // Sparse inference must accumulate in the same order for color-swapped
  // positions.  The representation is a set, so canonical sorting changes no
  // feature while making mirrored evaluation bit-for-bit reproducible.
  std::sort(result.begin(), result.begin() + count);
  return result;
}

float evaluate(const State &state) {
  if (state.terminal()) return -1.0F;
  return network().run(state);
}

SearchResult search(const State &state,
                    std::chrono::steady_clock::time_point deadline,
                    const SearchConfig &config) {
  SearchResult result;
  if (state.terminal()) return result;
  const bool timed = deadline != std::chrono::steady_clock::time_point::max();
  auto work_deadline = deadline;
  if (timed) {
    const auto now = std::chrono::steady_clock::now();
    constexpr auto kFinalizationReserve = std::chrono::milliseconds(6);
    work_deadline = deadline > now + kFinalizationReserve
        ? deadline - kFinalizationReserve : now;
  }
  const std::uint8_t root_player = state.to_move;
  std::vector<TreeNode> nodes;
  nodes.reserve(config.maximum_nodes);
  nodes.push_back(TreeNode{state});
  std::size_t iterations = 0;
  bool stop = false;
  auto generate_node = [&](TreeNode &node, bool root) {
    GeneratorStats stats;
    const auto started = std::chrono::steady_clock::now();
    node.actions = generate_actions(node.state, config.generator, root,
                                    &stats, work_deadline);
    const auto elapsed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - started).count());
    result.generator_microseconds += elapsed;
    result.maximum_generator_microseconds =
        std::max(result.maximum_generator_microseconds, elapsed);
    if (stats.deadline_reached) {
      ++result.generator_deadline_stops;
      result.deadline_reached = true;
      stop = true;
    }
  };
  while (nodes.size() < config.maximum_nodes &&
         std::chrono::steady_clock::now() < work_deadline && !stop) {
    int node_index = 0;
    std::vector<int> path{0};
    while (true) {
      TreeNode &node = nodes[node_index];
      if (node.state.terminal()) break;
      if (!node.generated) {
        generate_node(node, node_index == 0);
        node.children.assign(node.actions.size(), -1);
        node.generated = true;
        if (node_index == 0) result.root_actions = node.actions.size();
        if (stop) break;
      }
      if (node.actions.empty()) break;
      const std::size_t allowed = std::min<std::size_t>(
          node.actions.size(), 1 + static_cast<std::size_t>(
              std::sqrt(static_cast<double>(node.visits + 1) * 2.0)));
      if (node.expanded < allowed) {
        const std::size_t action_index = node.expanded++;
        State child_state = node.state;
        if (!apply_action(child_state, node.actions[action_index])) continue;
        const int child_index = static_cast<int>(nodes.size());
        nodes.push_back(TreeNode{child_state, node_index,
                                 static_cast<int>(action_index)});
        nodes[node_index].children[action_index] = child_index;
        node_index = child_index;
        path.push_back(node_index);
        break;
      }
      int selected = -1;
      double selected_score = -std::numeric_limits<double>::infinity();
      const double parent_log = std::log(static_cast<double>(node.visits) + 2.0);
      const double sign = node.state.to_move == root_player ? 1.0 : -1.0;
      for (std::size_t child_slot = 0; child_slot < allowed; ++child_slot) {
        const int child_index = node.children[child_slot];
        if (child_index < 0) continue;
        const TreeNode &child = nodes[child_index];
        const double mean = child.visits == 0 ? 0.0
                                             : child.total / child.visits;
        const double bonus = config.exploration *
            std::sqrt(parent_log / (static_cast<double>(child.visits) + 1.0));
        const double score = sign * mean + bonus;
        if (score > selected_score) {
          selected_score = score;
          selected = child_index;
        }
      }
      if (selected < 0) break;
      node_index = selected;
      path.push_back(node_index);
    }
    const float value = root_value(nodes[node_index].state, root_player);
    for (const int visited : path) {
      ++nodes[visited].visits;
      nodes[visited].total += value;
    }
    ++iterations;
    if ((iterations & 31U) == 0 &&
        std::chrono::steady_clock::now() >= work_deadline) {
      result.deadline_reached = true;
      break;
    }
  }
  result.nodes = nodes.size();
  if (!nodes[0].generated) {
    generate_node(nodes[0], true);
    nodes[0].children.assign(nodes[0].actions.size(), -1);
    nodes[0].generated = true;
    result.root_actions = nodes[0].actions.size();
  }
  if (nodes[0].actions.empty()) return result;
  std::size_t best = 0;
  std::uint32_t best_visits = 0;
  double best_mean = -std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < nodes[0].actions.size(); ++index) {
    if (timed && index != 0 &&
        std::chrono::steady_clock::now() + std::chrono::milliseconds(1) >=
            deadline) {
      result.deadline_reached = true;
      break;
    }
    State successor = state;
    apply_action(successor, nodes[0].actions[index]);
    if (terminal_for_mover(successor, root_player)) {
      result.action = nodes[0].actions[index];
      result.root_value = 1.0;
      return result;
    }
    const int child_index = nodes[0].children[index];
    const std::uint32_t visits = child_index < 0 ? 0 : nodes[child_index].visits;
    const double mean = child_index < 0 || visits == 0
        ? root_value(successor, root_player)
        : nodes[child_index].total / visits;
    if (visits > best_visits || (visits == best_visits && mean > best_mean)) {
      best = index;
      best_visits = visits;
      best_mean = mean;
    }
  }
  result.action = nodes[0].actions[best];
  result.root_value = best_mean;
  return result;
}

State rotate_and_swap(const State &state) {
  State result = state;
  result.used.fill(0);
  const auto &topology = Topology::get();
  for (int edge = 0; edge < kEdgeCount; ++edge) {
    if (edge_used(state, edge)) {
      const int rotated = topology.rotated_edge(edge);
      result.used[rotated >> 6] |= 1ULL << (rotated & 63);
    }
  }
  result.ball = static_cast<std::uint8_t>(topology.rotated_vertex(state.ball));
  result.to_move = static_cast<std::uint8_t>(1 - state.to_move);
  if (state.winner >= 0) result.winner = static_cast<std::int8_t>(1 - state.winner);
  return result;
}

}  // namespace jacek_arena_bfm
