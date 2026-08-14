
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace papersoccer {

enum class Player { One, Two };

enum class Status { InProgress, WonByOne, WonByTwo };

enum class GoalRule { OpponentGoalOnly, OwnGoalsAllowed };

enum class BlockedRule { PlayerToMoveLoses, MoverLoses };

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
GoalRule goal_rule{GoalRule::OpponentGoalOnly};
BlockedRule blocked_rule{BlockedRule::PlayerToMoveLoses};
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

namespace papersoccer {

bool is_regular_point(const RulesConfig &config, Point point);

bool is_goal_point(const RulesConfig &config, Point point);

bool is_attacking_goal(const RulesConfig &config, Point point, Player player);

bool is_boundary_point(const RulesConfig &config, Point point);

bool is_neighbor(Point from, Point to);

bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment);

std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player);

}  // namespace papersoccer

namespace papersoccer {

GameState make_initial_state(const RulesConfig &config = {});

std::vector<Move> legal_moves(const GameState &state);

bool grants_extra_turn(const GameState &before, Point destination);

GameState apply_move(const GameState &state, Move move);

bool is_terminal(const GameState &state);

std::optional<Player> winner(const GameState &state);

}  // namespace papersoccer

namespace papersoccer::detail {

inline constexpr std::size_t kMaximumMoves = 8;

inline constexpr bool same_rules_config(const RulesConfig &lhs,
const RulesConfig &rhs) noexcept {
return lhs.width == rhs.width && lhs.height == rhs.height &&
lhs.goal_rule == rhs.goal_rule &&
lhs.blocked_rule == rhs.blocked_rule;
}

struct PositionKey {
std::uint64_t first{};
std::uint64_t second{};

constexpr bool operator==(const PositionKey &) const noexcept = default;
};

inline constexpr std::uint64_t mix_position_key(std::uint64_t value) noexcept {
value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
return value ^ (value >> 31U);
}

inline constexpr PositionKey position_key_component(std::uint64_t category,
std::uint64_t index) noexcept {
const std::uint64_t tagged = (category << 56U) ^ index;
return PositionKey{
mix_position_key(tagged ^ 0x243f6a8885a308d3ULL),
mix_position_key(tagged ^ 0x13198a2e03707344ULL),
};
}

inline constexpr void xor_position_key(PositionKey &target,
PositionKey value) noexcept {
target.first ^= value.first;
target.second ^= value.second;
}

class CompactBitset {
public:
CompactBitset() = default;
explicit CompactBitset(std::size_t bit_count)
: words_((bit_count + 63U) / 64U) {}

bool test(std::size_t index) const noexcept {
return (words_[index / 64U] & (std::uint64_t{1} << (index % 64U))) != 0;
}

void set(std::size_t index) noexcept {
words_[index / 64U] |= std::uint64_t{1} << (index % 64U);
}

void reset(std::size_t index) noexcept {
words_[index / 64U] &= ~(std::uint64_t{1} << (index % 64U));
}

bool operator==(const CompactBitset &) const noexcept = default;

private:
std::vector<std::uint64_t> words_{};
};

class SearchTopology {
public:
using VertexIndex = std::uint32_t;
using EdgeIndex = std::uint32_t;

struct Arc {
VertexIndex destination{};
EdgeIndex edge{};
Move move{};
};

struct Adjacency {
std::array<Arc, kMaximumMoves> arcs{};
std::uint8_t count{};
};

explicit SearchTopology(RulesConfig config) : config_(config) {
if (config.width < 1 || config.height < 1) {
throw std::invalid_argument("search requires positive board dimensions");
}

const std::size_t regular_vertex_count =
static_cast<std::size_t>(config.width + 1) *
static_cast<std::size_t>(config.height + 1);
points_.reserve(regular_vertex_count + 6U);
point_indices_.reserve(regular_vertex_count + 6U);

for (int y = 1; y <= config.height + 1; ++y) {
for (int x = 0; x <= config.width; ++x) {
add_vertex(Point{x, y});
}
}

const int center_x = config.width / 2;
for (int x = center_x - 1; x <= center_x + 1; ++x) {
add_vertex(Point{x, 0});
add_vertex(Point{x, config.height + 2});
}

player_one_adjacency_.resize(points_.size());
player_two_adjacency_.resize(points_.size());

edge_indices_.reserve(regular_vertex_count * 4U);
for (VertexIndex vertex = 0; vertex < points_.size(); ++vertex) {
if (!is_regular_point(config_, points_[vertex])) {
continue;
}
build_adjacency(vertex, Player::One, player_one_adjacency_[vertex]);
build_adjacency(vertex, Player::Two, player_two_adjacency_[vertex]);
}
}

const RulesConfig &config() const noexcept { return config_; }
std::size_t vertex_count() const noexcept { return points_.size(); }
std::size_t edge_count() const noexcept { return edge_indices_.size(); }

std::optional<VertexIndex> find_vertex(Point point) const noexcept {
const auto found = point_indices_.find(point);
if (found == point_indices_.end()) {
return std::nullopt;
}
return found->second;
}

std::optional<EdgeIndex> find_edge(const Segment &segment) const noexcept {
const auto found = edge_indices_.find(segment);
if (found == edge_indices_.end()) {
return std::nullopt;
}
return found->second;
}

Point point(VertexIndex vertex) const noexcept { return points_[vertex]; }

const Adjacency &adjacency(VertexIndex vertex, Player player) const noexcept {
return player == Player::One ? player_one_adjacency_[vertex]
: player_two_adjacency_[vertex];
}

private:
RulesConfig config_{};
std::vector<Point> points_{};
std::unordered_map<Point, VertexIndex, PointHash> point_indices_{};
std::unordered_map<Segment, EdgeIndex, SegmentHash> edge_indices_{};
std::vector<Adjacency> player_one_adjacency_{};
std::vector<Adjacency> player_two_adjacency_{};

void add_vertex(Point point) {
if (point_indices_.contains(point)) {
return;
}
const auto index = static_cast<VertexIndex>(points_.size());
points_.push_back(point);
point_indices_.emplace(point, index);
}

EdgeIndex edge_index(const Segment &segment) {
const auto existing = edge_indices_.find(segment);
if (existing != edge_indices_.end()) {
return existing->second;
}
const auto index = static_cast<EdgeIndex>(edge_indices_.size());
edge_indices_.emplace(segment, index);
return index;
}

void build_adjacency(VertexIndex source, Player player, Adjacency &result) {
const Point from = points_[source];
for (const Point destination : neighbors(config_, from, player)) {
const Segment segment{from, destination};
if (is_forbidden_boundary_segment(config_, segment)) {
continue;
}
const auto destination_index = find_vertex(destination);
if (!destination_index.has_value()) {
throw std::logic_error("search topology is missing a legal destination");
}
if (result.count >= kMaximumMoves) {
throw std::logic_error("paper soccer vertex has more than eight moves");
}
result.arcs[result.count++] =
Arc{*destination_index, edge_index(segment), Move{destination}};
}
}
};

class SearchPosition {
public:
using VertexIndex = SearchTopology::VertexIndex;
using EdgeIndex = SearchTopology::EdgeIndex;

struct Undo {
VertexIndex previous_ball{};
EdgeIndex edge{};
Player previous_player{Player::One};
Status previous_status{Status::InProgress};
bool destination_was_visited{};
};

SearchPosition(std::shared_ptr<const SearchTopology> topology,
const GameState &state)
: topology_(std::move(topology)), used_edges_(topology_->edge_count()),
visited_vertices_(topology_->vertex_count()), to_move_(state.to_move),
status_(state.status) {
if (!same_rules_config(topology_->config(), state.config)) {
throw std::invalid_argument("search topology does not match game state");
}
const auto ball = topology_->find_vertex(state.ball);
if (!ball.has_value()) {
throw std::invalid_argument("search game state ball is outside the board");
}
ball_ = *ball;

for (const Segment &segment : state.used_segments) {
const auto edge = topology_->find_edge(segment);
if (edge.has_value()) {
used_edges_.set(*edge);
xor_position_key(position_key_, position_key_component(1, *edge));
}
}
for (const auto &[point, visits] : state.visit_count) {
if (visits <= 0) {
continue;
}
const auto vertex = topology_->find_vertex(point);
if (vertex.has_value()) {
visited_vertices_.set(*vertex);
xor_position_key(position_key_, position_key_component(2, *vertex));
}
}
add_dynamic_key();
undo_stack_.reserve(topology_->edge_count());
}

const std::shared_ptr<const SearchTopology> &topology() const noexcept {
return topology_;
}
Point ball() const noexcept { return topology_->point(ball_); }
VertexIndex ball_vertex() const noexcept { return ball_; }
Player to_move() const noexcept { return to_move_; }
Status status() const noexcept { return status_; }
bool is_terminal() const noexcept { return status_ != Status::InProgress; }
std::size_t undo_depth() const noexcept { return undo_stack_.size(); }
PositionKey position_key() const noexcept { return position_key_; }

bool edge_used(EdgeIndex edge) const noexcept { return used_edges_.test(edge); }
bool vertex_visited(VertexIndex vertex) const noexcept {
return visited_vertices_.test(vertex);
}

bool grants_extra_turn(std::uint8_t slot) const noexcept {
const SearchTopology::Arc &arc =
topology_->adjacency(ball_, to_move_).arcs[slot];
return is_boundary_point(topology_->config(),
topology_->point(arc.destination)) ||
visited_vertices_.test(arc.destination);
}

std::optional<Player> winner() const noexcept {
if (status_ == Status::WonByOne) {
return Player::One;
}
if (status_ == Status::WonByTwo) {
return Player::Two;
}
return std::nullopt;
}

std::uint8_t legal_slots(
std::array<std::uint8_t, kMaximumMoves> &slots) const noexcept {
if (is_terminal()) {
return 0;
}
const SearchTopology::Adjacency &adjacency =
topology_->adjacency(ball_, to_move_);
std::uint8_t count = 0;
for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
if (!used_edges_.test(adjacency.arcs[slot].edge)) {
slots[count++] = slot;
}
}
return count;
}

Move move_for_slot(std::uint8_t slot) const noexcept {
return topology_->adjacency(ball_, to_move_).arcs[slot].move;
}

bool slot_for_move(Move move, std::uint8_t &result) const noexcept {
const SearchTopology::Adjacency &adjacency =
topology_->adjacency(ball_, to_move_);
for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
if (adjacency.arcs[slot].move == move &&
!used_edges_.test(adjacency.arcs[slot].edge)) {
result = slot;
return true;
}
}
return false;
}

void make_move(std::uint8_t slot) {
if (is_terminal()) {
throw std::invalid_argument("cannot move a terminal search position");
}
const SearchTopology::Adjacency &adjacency =
topology_->adjacency(ball_, to_move_);
if (slot >= adjacency.count || used_edges_.test(adjacency.arcs[slot].edge)) {
throw std::invalid_argument("illegal compact search move");
}

const SearchTopology::Arc arc = adjacency.arcs[slot];
const Player mover = to_move_;
const bool destination_was_visited = visited_vertices_.test(arc.destination);
undo_stack_.push_back(
Undo{ball_, arc.edge, to_move_, status_, destination_was_visited});
remove_dynamic_key();
used_edges_.set(arc.edge);
xor_position_key(position_key_, position_key_component(1, arc.edge));
visited_vertices_.set(arc.destination);
if (!destination_was_visited) {
xor_position_key(position_key_,
position_key_component(2, arc.destination));
}
ball_ = arc.destination;

const Point destination = topology_->point(ball_);
if (is_goal_point(topology_->config(), destination)) {
status_ =
is_attacking_goal(topology_->config(), destination, Player::One)
? Status::WonByOne
: Status::WonByTwo;
add_dynamic_key();
return;
}

const bool extra_turn =
is_boundary_point(topology_->config(), destination) ||
destination_was_visited;
if (!extra_turn) {
to_move_ = opponent(to_move_);
}
status_ = Status::InProgress;

std::array<std::uint8_t, kMaximumMoves> slots{};
if (legal_slots(slots) == 0) {
const Player blocked_player =
topology_->config().blocked_rule == BlockedRule::MoverLoses
? mover
: to_move_;
status_ = blocked_player == Player::One ? Status::WonByTwo
: Status::WonByOne;
}
add_dynamic_key();
}

void unmake_move() {
if (undo_stack_.empty()) {
throw std::logic_error("cannot unmake the compact search root");
}
const Undo undo = undo_stack_.back();
undo_stack_.pop_back();

const VertexIndex destination = ball_;
remove_dynamic_key();
ball_ = undo.previous_ball;
to_move_ = undo.previous_player;
status_ = undo.previous_status;
used_edges_.reset(undo.edge);
xor_position_key(position_key_, position_key_component(1, undo.edge));
if (!undo.destination_was_visited) {
visited_vertices_.reset(destination);
xor_position_key(position_key_, position_key_component(2, destination));
}
add_dynamic_key();
}

void unmake_to(std::size_t depth) {
while (undo_stack_.size() > depth) {
unmake_move();
}
}

bool same_compact_state(const SearchPosition &other) const noexcept {
return same_rules_config(topology_->config(), other.topology_->config()) &&
ball_ == other.ball_ && to_move_ == other.to_move_ &&
status_ == other.status_ && used_edges_ == other.used_edges_ &&
visited_vertices_ == other.visited_vertices_;
}

private:
std::shared_ptr<const SearchTopology> topology_{};
CompactBitset used_edges_{};
CompactBitset visited_vertices_{};
VertexIndex ball_{};
Player to_move_{Player::One};
Status status_{Status::InProgress};
std::vector<Undo> undo_stack_{};
PositionKey position_key_{};

void add_dynamic_key() noexcept {
xor_position_key(position_key_, position_key_component(3, ball_));
xor_position_key(position_key_,
position_key_component(4, static_cast<std::uint64_t>(to_move_)));
xor_position_key(position_key_,
position_key_component(5, static_cast<std::uint64_t>(status_)));
}

void remove_dynamic_key() noexcept { add_dynamic_key(); }
};

struct TacticalProbeStats {
std::uint64_t nodes{};
std::uint32_t max_depth{};
bool depth_cutoff{};
bool node_cutoff{};
};

struct TacticalProbeResult {
std::optional<Player> proven_winner{};
std::optional<Move> proving_move{};
TacticalProbeStats stats{};
};

TacticalProbeResult run_tactical_probe(SearchPosition &position,
std::uint32_t max_depth,
std::uint32_t max_nodes);

}  // namespace papersoccer::detail

namespace papersoccer {

namespace {

constexpr int kNorthGoalY = 0;
constexpr int kFieldTopY = 1;

int field_bottom_y(const RulesConfig &config) { return config.height + 1; }

int south_goal_y(const RulesConfig &config) { return config.height + 2; }

int center_x(const RulesConfig &config) { return config.width / 2; }

int mouth_left_x(const RulesConfig &config) { return center_x(config) - 1; }

int mouth_right_x(const RulesConfig &config) { return center_x(config) + 1; }

bool is_mouth_x(const RulesConfig &config, int x) {
return x >= mouth_left_x(config) && x <= mouth_right_x(config);
}

bool is_north_goal(const RulesConfig &config, Point point) {
return is_mouth_x(config, point.x) && point.y == kNorthGoalY;
}

bool is_south_goal(const RulesConfig &config, Point point) {
return is_mouth_x(config, point.x) && point.y == south_goal_y(config);
}

bool is_goal_mouth_point(const RulesConfig &config, Point point) {
return is_mouth_x(config, point.x) &&
(point.y == kFieldTopY || point.y == field_bottom_y(config));
}

bool is_north_goal_post_segment(const RulesConfig &config, Segment segment) {
const bool touches_north_goal = is_north_goal(config, segment.a) || is_north_goal(config, segment.b);
if (!touches_north_goal) {
return false;
}

return segment.a.x == segment.b.x &&
(segment.a.x == mouth_left_x(config) || segment.a.x == mouth_right_x(config)) &&
((segment.a.y == kNorthGoalY && segment.b.y == kFieldTopY) ||
(segment.a.y == kFieldTopY && segment.b.y == kNorthGoalY));
}

bool is_south_goal_post_segment(const RulesConfig &config, Segment segment) {
const bool touches_south_goal = is_south_goal(config, segment.a) || is_south_goal(config, segment.b);
if (!touches_south_goal) {
return false;
}

return segment.a.x == segment.b.x &&
(segment.a.x == mouth_left_x(config) || segment.a.x == mouth_right_x(config)) &&
((segment.a.y == field_bottom_y(config) && segment.b.y == south_goal_y(config)) ||
(segment.a.y == south_goal_y(config) && segment.b.y == field_bottom_y(config)));
}

}  // namespace

bool is_regular_point(const RulesConfig &config, Point point) {
return point.x >= 0 && point.x <= config.width && point.y >= kFieldTopY &&
point.y <= field_bottom_y(config);
}

bool is_goal_point(const RulesConfig &config, Point point) {
return is_north_goal(config, point) || is_south_goal(config, point);
}

bool is_attacking_goal(const RulesConfig &config, Point point, Player player) {
return player == Player::One ? is_north_goal(config, point)
: is_south_goal(config, point);
}

bool is_boundary_point(const RulesConfig &config, Point point) {
if (!is_regular_point(config, point)) {
return false;
}

if (point.x == 0 || point.x == config.width) {
return true;
}

const bool on_goal_line =
point.y == kFieldTopY || point.y == field_bottom_y(config);
const bool inside_goal_opening =
point.x > mouth_left_x(config) && point.x < mouth_right_x(config);
return on_goal_line && !inside_goal_opening;
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

if (segment.a.y == segment.b.y &&
(segment.a.y == kFieldTopY || segment.a.y == field_bottom_y(config)) &&
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
const bool own_goals_allowed =
config.goal_rule == GoalRule::OwnGoalsAllowed;
if ((own_goals_allowed || player == Player::One) &&
from.y == kFieldTopY) {
for (int goal_x = mouth_left_x(config); goal_x <= mouth_right_x(config); ++goal_x) {
const Point goal_point{goal_x, kNorthGoalY};
if (is_neighbor(from, goal_point)) {
result.push_back(goal_point);
}
}
}
if ((own_goals_allowed || player == Player::Two) &&
from.y == field_bottom_y(config)) {
for (int goal_x = mouth_left_x(config); goal_x <= mouth_right_x(config); ++goal_x) {
const Point goal_point{goal_x, south_goal_y(config)};
if (is_neighbor(from, goal_point)) {
result.push_back(goal_point);
}
}
}
}

return result;
}

}  // namespace papersoccer

namespace papersoccer {

GameState make_initial_state(const RulesConfig &config) {
GameState state;
state.config = config;
state.ball = Point{config.width / 2, config.height / 2 + 1};
state.to_move = Player::One;
state.status = Status::InProgress;
state.path = {state.ball};
state.visit_count[state.ball] = 1;
return state;
}

std::vector<Move> legal_moves(const GameState &state) {
std::vector<Move> result;
if (state.status != Status::InProgress) {
return result;
}

for (const Point destination : neighbors(state.config, state.ball, state.to_move)) {
const Segment segment{state.ball, destination};
if (state.used_segments.find(segment) != state.used_segments.end()) {
continue;
}
if (is_forbidden_boundary_segment(state.config, segment)) {
continue;
}
result.push_back(Move{destination});
}

return result;
}

bool grants_extra_turn(const GameState &before, Point destination) {
if (is_boundary_point(before.config, destination)) {
return true;
}
const auto it = before.visit_count.find(destination);
return it != before.visit_count.end() && it->second > 0;
}

GameState apply_move(const GameState &state, Move move) {
if (state.status != Status::InProgress) {
throw std::invalid_argument("cannot apply move to terminal game state");
}

const auto moves = legal_moves(state);
const auto legal_it = std::find_if(
moves.begin(), moves.end(),
[&](const Move &candidate) { return candidate.to == move.to; });
if (legal_it == moves.end()) {
throw std::invalid_argument("illegal move");
}

GameState next = state;
next.used_segments.insert(Segment{state.ball, move.to});
next.ball = move.to;
next.path.push_back(move.to);
next.visit_count[move.to] += 1;

if (is_goal_point(next.config, move.to)) {
next.status = is_attacking_goal(next.config, move.to, Player::One)
? Status::WonByOne
: Status::WonByTwo;
return next;
}

const bool extra_turn = grants_extra_turn(state, move.to);
next.to_move = extra_turn ? state.to_move : opponent(state.to_move);
next.status = Status::InProgress;

if (legal_moves(next).empty()) {
const Player blocked_player =
next.config.blocked_rule == BlockedRule::MoverLoses ? state.to_move
: next.to_move;
next.status = blocked_player == Player::One ? Status::WonByTwo
: Status::WonByOne;
}

return next;
}

bool is_terminal(const GameState &state) {
return state.status != Status::InProgress;
}

std::optional<Player> winner(const GameState &state) {
if (state.status == Status::WonByOne) {
return Player::One;
}
if (state.status == Status::WonByTwo) {
return Player::Two;
}
return std::nullopt;
}

}  // namespace papersoccer

namespace papersoccer::turn_action_v2::replay_book {

struct Replay {
std::uint8_t player_id;
std::uint8_t first_turn;
std::string_view transcript;
};

inline constexpr std::array<Replay, 24> kReplays{{
{0, 6, "0/5/21/2/7/523/3/35/0/36350/07/5/6/6/0/5/5/631/16430/5023/21/41674/7577/2/21/3/577/55327/724101305/533/3050100/321647/66666/72/5202/243/11425056542006074772/71/714/3/66144/3/6302/330/007/2/2/2/0/0363636/7507/13/34546764750014/71/05211"},
{1, 5, "1/1/7/6/6/3/07/05/74/53/1/2060350634/14/663/1211/4/1/03635/6060/1474553/032/17723/364/27056/56/30206577/724452/22076524144/2/0574/1313/12744/3/17563/235/350/17/1647074665/3/5/2/01/05/0367635/61/7241466335/6300/61420/742164203/02/3/24705/652/105633/6/35"},
{0, 4, "0/6/5/5/71/34/1/033/5/5/2/5/57/1/2477/1647253/463071420/33/641000/7245/7470/063224/366111/3/0/3/17/55/520061/6/7/2/1/03/057/65/753/0/14652530/1/21/02543367432235/36/17/02545/2357/571/32/05417667070/20/1465317074612/03632/025441/1/427713527/46/360633/45/64/47230/00531657667161/65"},
{0, 2, "0/2/7/45/7/5/71/34/2212/2/7/1/6/1/03636074/33535/7/00535/0/633/10/35/035/27635/50/241/035067/46/57/1/0/5/2310/256/102/16613165475/5331/17574/236746167/064/276313/17/60/672/3310/6/52/42756/06310/50223611/07"},
{1, 3, "1/1/0/1/275/03674/2476/4/7/75/6143/123/617/505/3/23/6/125032/0/361745642/4/1/63/2774142564/2/03677770645/72/520224/6/03144/75/0653/13/6/4/74/61632116/50275/3552/14/7421/02716443/1/175364613/272564/3/1/756/035/725/6/614145/2/500333"},
{1, 3, "7/6/7/53/10/34/71/45/221/2/1/35/70/54/17/43/660/33/020/641364/2755/2/166/1336/74747/5/0/5/6/3/5/31/7167/7223036135/24/3/50/241/25727/0252/75767/234/4/125635"},
{0, 2, "0/1/0/6/33/5/3/6/5/3/6/05/21/66/7/4/6/3/161/13442/72/01/17/0/255336/5702/4202577/22/520167/5452066/6/1/7/7/0524/53/27/164453/2572461330067/2/50141016/64/146652134/632/0613574302100/164721/41675/03/5030654/47271/17"},
{1, 19, "6/7/5/61/44/53/0/0617227/47271/053/03/5023/1/4/7/425/61434/3/0/3/65/3/41/65/67/1/6/47275253147/11/425/01/61410/63613/42/754/607/03647/16/366163071/123/4/555/234/44"},
{1, 3, "3/1/7/7/7/2/2/505/6/6/1/7/5352/7243/2/3/6/02574/5/3/6/6171/6110300/5253/312/1/7/2/0/25252744/64136/02566/44/71134/4/27254/4/166/74/6/16/323257/50/1467016023/03663/534705/07/2050610164563353/57/006353012/3647/23/02525"},
{0, 40, "0/3/2/3/6/5/2/4/76/5/6/4/10/55/2476317/1/221/030524745/674663071/36/071/3/0/2/21/3/5671/47/5671/4/72714/43/65212221/4/2/505/247021/2506335/2553/66/677/0174744/366311/1256650134/6413312/005325357/014271761/0270/56/57/47/22/2470/13/030522575771"},
{0, 18, "0/5/21/3/0/3/17/55/061/4276/1/42/217/4/427570/5/4432225/35/36/57/1/4/125664130061/02745/2470/274763/25/5/0/5/230/65/57/4721/2470/5/71/335/7242476061/721/0/364/671/33/0523111/60575/6671/4/721/7533/6301/74/2221/3166614743/67/756/3552133550500/33/6631231642706/427543/07027027/4561160327417/01"},
{1, 3, "7/6/1/44/21/7/7/422/7/455/00/21657/1/75353/10/27433/141/3/17/25/42700/256/3607/135524/665760/166352325/06/33/0/10676356144/133/3/1/6/1/44/2/71/7/42355/01/435755"},
{0, 20, "7/6/0/35/01/44/21/4/1/63/07/2/57/25/052761/421/1/4/1/7474/42474176"},
{0, 4, "0/6/5/4/5/53/61/72442/30/2/7/46413/00/2/531/316/50216/56/52076572/450120/357177/532/21/03/2/3/2500/65/441/24763072/4167474/2507/2253117/50/247656431605470/7272/350230/270/330/1/7/2/42/25/2743527/677/45017/2/46124/5201317/05/74/53611631757/65/31/30616474/60317"},
{1, 3, "1/1/7/6/0/75/74/3/00523/135/01/13/27435/35/7/164675/6/6/71/72524/4611232/764/763/30654/17201/3550102144444/1721/425/46/16467/5/024674/75252/53/530/1/0/711324/75/0632/3/1/3/17505675/23/2027545/2/13/3/25050/50/11356675/47/4772420613520633/10/3067425/46165"},
{0, 4, "0/6/5/4/5/3/6/7243/5716172/23/0/3/1/2/1/35/77/43/225/4/250174700/54/661/45/05417/60316443/47411/4/60217077/4/7/53/120/543212225/017/4/167/1/75/32/036175/74235/357/41/652770/55/44717271/4/71413/5/2001"},
{1, 1, "4/6/6/3/6/3/2/746/6/14117/7/7/71/63/003/14/6544/61613/0/7144524612/4/61713/4/53550143/01/6605/74/20553/6301021/72432/4/2/75/555/7531070/723613423/2/4/71/43/17/0522725743/425750/306064/5747/72422/13610/722744641320555/0603364"},
{1, 1, "2/4/6/05/0/33/6/7241767/1/1/6/42/5/67/63/610/2/5453/2/10077/46114/360753114/54/1303/63/03/6455/0636/74/205721/0613316/6521312/7/43/163635/27/000/6657645/3074453/63/6113/024/11/247642/71/25721165444235/4167/03/056/666/52/06333"},
{0, 30, "0/3/2/3/6/5/2/4/76/5/6/4/10/55/2476317/1/221/030524745/674663071/36/071/3/0/2/21/3/0/3/17/25/0567/4/7/5/4610/6/1/36075333325/4471271/065556333/63012/5/31/0632/25071/01703664723/052531636650706/66/063131241160/1/2566/357165/247020/756/46131/213/0305247575001"},
{1, 7, "1/1/7/6/0/75/03/213/570/6433/00/063203/0365/4644/1/3/0/2535/0/35/07/216575/022/003545/77/6/7/53/0/5253/0/5252/01/2/7/43/00/164561613255/074321/7672410/1/013563335/06/52032366/03350/57/41/65/67/614474113/0/3644"},
{1, 11, "0/0/0/6/6/5/2/5/0527271/614347/013/433/1/3/17/54/657/5/2/5/06/61434/7/23/1/7244/1/2/1/164276336/60/335/7/4/225/41675/41/66/57/7/7101/4722223535/242570310/052746/063450/166652/06/45310136/00/35335"},
{0, 20, "0/3/2/3/6/5/2/4/76/5/6/5/63072/1/30/24/764671/1/0/2/21/3/44710/3/17/25/0527/1/03560/03475/24607/4/7/0365/0/031/0"},
{1, 19, "7/6/0/2/1/4/1/2/0/53/01/2750363/03667/274466/0/75/0522/205335023/0/2554/1/65/7/0105354/7/53/11/6613425/0632/5/06/653/0/063135/752250300/23/31/30/7/447/10/3/17/55/775470/071/0/05327524/10/13313/0525/5023/250367/63/00707/4165633/025641231/274455/74236522702024/67/07463254/453"},
{1, 3, "0/0/3/67/27/45/5/2/5/6143/5/717271/1/7/532/27412/41/654/74761/06344/1721/05/3075221/23/57/063633/652141/6033/35/6/3/35/27/002525457/41/65/67/57132/1/31/2530606/1/6/7/012/216/36016/17535254/35475206/7/713550/074322446/547/71/0252347/460220/64117/0642447123/300561/4535/502363"},
}};

}  // namespace papersoccer::turn_action_v2::replay_book

namespace papersoccer::turn_action_v2::replay_value_model {

inline constexpr std::size_t kInputCount = 1156;
inline constexpr std::size_t kEdgeCount = 316;
inline constexpr std::size_t kVertexCount = 105;
inline constexpr std::size_t kHiddenOne = 8;
inline constexpr std::size_t kHiddenTwo = 8;
inline constexpr float kW1Scale = 0.00294888325F;
inline constexpr float kW2Scale = 0.0120762014F;
inline constexpr float kW3Scale = 0.00617757604F;
inline constexpr std::array<float, 8> kB1{{0.0799908191F, 0.0400607735F, 0.0325085968F, 0.0263906606F, 0.0389416032F, 0.0440370515F, 0.0529267117F, 0.0267492924F}};
inline constexpr std::array<float, 8> kB2{{0.0700218901F, -0.0561930127F, -0.0431844927F, -0.0496364459F, -0.0308015831F, -0.0599980950F, -0.0358039625F, 0.0284082294F}};
inline constexpr float kB3 = 0.0647875145F;
inline constexpr std::string_view kEncodedWeights = "AAAAAAAAAAAKDfYU8A8GCv7xEgv38w0BA/El7hoNGiX86gYf5gH/FvTSBA8P8woV79DyEgzpEwX/4xH7HuUk9/bdIP4R7v/t9OcHCg7xJif89wwG8xX0CQAAAAAAAAAA9gsL9QEIECL6/uEa9QUG8vT8GuYa+iAcAuL0GvEHCgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAUHNkPwxDhD/jI8gwTBRQcIu8BIP34Eg0AAAAAAAAAAAzQAwIZ/gcBCeUG+xUCHPHv1Qb3/+8EAd76Me4D6S0e/cQcAxfrChQIsjL7GvwpHAAAAAAAAAAA8r0k/CjlIP/52CUKE98KGPT+FxcDDRsTB+QUFRYC+Bfv5wALAvMLBd/aFf8x7RMc+dsn+iHhCSnv+BMVBOoRGvLXEv4X6jIF+d0L7AfuAgbu4CwIFdUrEfMFCwXkC+j7CQ3+AQMDAAX11/kF9OgULe65DAAQ8hr0EdwMDgEODBj66xIML/MZFQPR/PsPCgP5+xkKCwEO/PjxEgz9AywNCfPsJAQL9hUN+fP99AQD+DIM5e4OCxcAC/vsLQ0A+vT8BtoZAyfsBRz36vIFLwL+JwvZAxs20BYp88Yd90HcJR3U5RkBLdkkGufLFAAO8Qn8BNv2CeIpDQAKyQQa/PcQFejcFgAv5RIQEDQMH/oO+fvo1iv2IvkRHRUG7i3KG/rf8KINFRfNEicH4Bz8IcQpFvj99hAZAwTk8TcEEc04+uMCwiTuMeQiMvsP8Q3vKhTyDPz8Hvb6CvbaswAGGeIrFvXFFvgk8A0Y5MsM/Bv1Ew3WlyjzMsBCRRrPJfwY1SINCNgx4T3HIyjaryf0L+UyKjtA4wjWOfHnGirqLQgfAd/zzQkUFf4V+fPhGAQO8RwXEwXjF+IZ4ekL+wwm8AgF5hUK8QUCEQMS8+/7EPoY+fPk6CgLA9URG/LdKf0ICvb1H/MQDfwQBv367iAVI+oO9vvfIBkS6hgNJRz0E+4c6/bv0wT79AcLAeDBEQgL5xQn+woa/hj1FRkH1AMW+vz7JQ1F8x3zMwHU/9L/LREaGhD32w4OIPMIERMJ5/rvG+/9+bsi/x4CIxfhqj8PNN8gKwPsOA4OBgABKTjjDeMW9tzx5wz8AusJJe2eN+0+2TIk7rMlHTv7GgDs2BkPD+MfDjxLyxCtO8fJHhAHG9ocAPX2vwkDOdgcIRw+AQznMgbc4MwN7DPTHvkBCfn4FvAgFgbaEegatxAr6uEcHRrlJibv6v0SCgohGQ/pJvIaAh0f2AcX8iriCwwE8hAfHu0aHurZHRUh9gARCd4QKBn7+hIH1gUQEQUK+e71HQEbDx8eEuQL+R4ODxAuRfgP/Q8N+fDOEQ5H3jZAAu32AxsCGwgD5SkDJgcQHf2vPvI0zyAyDOwlCTDxJSEsVa4hpS3jzg0zyRvNPuzcFUfqCNYz5dAUMQEqEwQWCyZhzwTATOK3LT3LCKY477sQM9VA1Tnc20hPxBS+Jeu5IyjhJO0I8twyNPwR3A4A+BkJ7f/tDwTmRlPZ/MA3+8wsHN8N4QwSCgjoMiYX9v8BC+kGFA3sHhoYC/4hAycS/wblFRsaBRoH8xIf/gMNHPzu1Sf/FekrIOoQDwoe2hEdIA3589Eo8ecJ5RkTIAAhGiTwCf7qIOjQEv4bAC71HQYk3BYd2QQD8BvYJgP2ABccFe/tEuv+EwoK//AYASH06hwt8wzSLxDyEQPoJtQJ6tr05BsgOeIxACX6DSvzAyMLGMEhATQRGiwN3vn3IgQeCP+eKQAl6BozD/QN6AwMCRrxpU7nRrhKSx3eIPs0IhUjFDL3FN80+vPSmFfWOLJAGROdIPwo+B0kE8klIAvlEx8EO+0H4xYC7gmHMQE1+j5GMTT4GdwQCOMi/AIpBxDy9yU41RHULPrLEP36BPQUGegd9f4T+hj55AUQ8wnd9gbuDBj6GQoX9QgMHOAX7yMD3xoJzyTOE/bnBhvxCvADFuYbMesd7SoM6wgC7AkIG+75IAnUEPUv9uwhKOgE5gjh8QcU2yjhDvTT6wr6HegKFu8U2CjuKegOIwu2OPpF/j08EErVD+El59YXDQELGvcaJu+jOwVV4EtCGxb/GNYTBer8ilfjOLtHTxZxtxW7NfnG/OM/7TbnDD8AuzUIQuAaU0Ms5xz4Lf/ZMjHM9tUg7OgYRLcjvyz44SDSCA4bCyURBiLSFc5H99oIJAcW5QPy6hjYHAAA9iAsLAj3Duch//sP3QAPJBgVDh0j4B39+fbTERL0AQIjIPwR0ygcNPwEAAYA9B74IA3vJe/0I/MM+8UJ/RQe9SUf9Aj8+Qj8FxPgFwbSCekfBd4fI7wi5w/n4wb6L/4Z+iQYDxoaBPcdGtga5/n5/QUK+xEx3v/3EgG9JQbr/vYWCucr/80V7w4M3RWBPgZF4U9QISvYDuIe4uMjJAwZ6SP12S5C4AnPJtbbHVLOGKcz1dQP7+8LBxUL1hUX6x/+BxnnURPo/esZ/e0Y5Qf2FgAWCRDX5xL7EAn9CyLZFNUq7u4c6gkN+xoa+w/6Cg4AKwrlF/nvDv8w5vcbMe/x2Bj56xcY6A/zJQzrHCzpF+AWDMsrLu0sxxMK/Ssl6hnfF/DeJdsi8ykUGAoX+wL/BDQZ5hwRBhL7CQjuGO0RFxoNDwVFFOcd6wAG7BAB8v4LDxIDIy/q+csUCs0SC+QGAgv83RcK+BrkLBIE+ukWEyQAGAUoB/gA6iTvBgz6JCP+ARjsCt0UAwINHgIdAfX82Q4A0iQxERb8IBL1DuDvCgkcCfQcGPga3wEG9Rjv5w/aK/7kEQbrGOoMAfIdLPYM6R4b1hf++in7DBTqDeQECybxKRcW9vAZ9hgG+SkF5PbpH+P7Gxn9FgQHEe8e8BYHDPcQGhMA2RzjBQzmLTjwIeEs9vQe+vgiAAsE5SUQ7PgHDR0HJB7gCuMYC+AcDQ8S8R4a7DEo3PbNCvTXIxbiCsolEtIAAAAAAAAAADI49//QHu3wFBnxFf8LEdoZIu8I6Bz6CSwWxfvYK/bfJw8JAvIHCOYYJvIgEQb9/AwC+RsX8f31EBT0Cusd6/ofB+QVGhUI7SU24f7qBQXeFAPrB/wVECMa7gYVFQ4IFh/g/gT/AQUFAQQjBvAD/PAK/AAABgUG/xcd7OrmCAbyIBzgA/Mq/uoWCQQT8gcTAhwZ+wH3Fh/3Hh0DD+EcFvEFCe0F+wsh6wcU+gz2F+v6JywDCOgM/OwvAv4Y6gQU9xYT8QfnLQoDCvXxCAgNA+wAAAAAAAAAACQFEvcVIw8UAxbeDP0C/fgAAAAAAAAAAB8U/+fvI//dAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAU6+MO5R/Z7AvoHgoZCBInB/0IAyAFIwsZH/YMFjgr/S0C/xb7/PrxGOEK/f/iGAgmF9kA9BwO+uYEH/4S8RYPEwPZHtIGCvcu+P4NBhQfEyYKCvoQFgghACn2BwHgD/4fICTt9AUQ6RcIBPEEBifYGQoDEv//Affc9hgHEfUSJALV+xUN9e0NFQQI9eYY7AsYEAr8FgoSDyQhF/z2IBUOGRkSKRj5+xEbENAEBBvtBjMU/PvhDgL8+gAg/hDvICMgAsQZtifRzBIB/QsWBgAGAuwZ9hMCFR4QKgIEJQgBAw0SBQwQ9AsIHxLYFwMQ/PcuF/EK6yfz0/EAFAMc9SQiMAHmJNMtzMgVAfoDBhsD+xL9CScYACj9GQsWBR/YGBkjIQIC9QQF5SIzEgfc7v0MI/sT/gQU9AXw7Rf8H/QXAhgCtyjLR+LFFfQAEPT/5A4gC/kJHRUTKxgG+wwm+yUnBxrZHeAH9OcX+Acc/AEh/AAHAgwaAREQAgASDSj0CxoQ9vX5CCYQEyPu8wrlIwEfJgfz+hsWGgwU+RfsCwITDhICD/r6Agn9JQr89/H+/+EIKv0h8QsO4Ov9AQAa6yIBEBTuCPYO4/Ibxg/59wT0BSMh8wYAD/P2Jib46Pv1CfgDBgALFu4IHSYT//ATDCAUKi7n/OQi7+v8+iL7C9sbBBnx/QL9+/YWLvb89+0F8/Ac4QIHIggaBB77HBMKCB4YEyYMBBH5DQUZLuvs9jAaCikSyhLcGhnc//gX9REW/Rw97uAQ0yoA7w/r+xgCF/gAFwEeEx3/KxoLAQgKH/gGIx8WEwQFBxj6NBD0BQUR9ecPFQQX//sG7ujy7u4Q3wL8EPP4GeAB5ucP4BMMLQIFLgcZC/YeAwUHFx7eAxwjCvMRBxgBEfEGES70EfstFiP+PCr5AOIFF+UKBN7oFfMIBh3f/xfu9PcPEfv6/BrRCAkeFxgRFPAX9xEYFAwdDxgTG/v/FQ0U8e8W+/MR6gsR+Ccs2xHVIO2/+Psl7AQU9/Me+/sj5Qzr2Afv9QHpChH0IicPKvQFCw0N9hPjHeY2CAXgMQoc5ykoEBfqA+kl4ecF//MHEBT95ecHGRAl3wwLKB3bHtMY2+8RAQIh9Qr2+h3wBwY06jASIBkLCSf6EPAa8QsK+AHzDxwXBQYKCe4D9gX7DhESCfgEARUGJ/AiCQ7a+fb45OP5Ew/yDvgICw4SCCXtGwIiDi8bBx8XAAcdB+0M+hD/Ix8DFAMl/hAl9An3+QAVCyr8DwgY9A/oAgoNB/MA9//77xfkAg0lGRcWKxoFF+caBPQaGwEP3xPyBRAg5gsDCPQEGwUV4vzvHAwR1iHrEfcTIewcEPcN7w4UMe3p9xQT7+skFv8E7hID/TAP7wUF/CDwGwH3CjIAFwgZAQz9AQIEBxwWAwAEFwD+IxTsK94VDan26hn77+kDBxLq9PMG+wQIGc8DEQroAws1Eg0F8BcXBx8G3hvzFAD5A+0PIBjYBhgWBQT8FgMRAgY2wxDKHfLTFPcP+SL1HP8OBu4K4RLx4BHy+AUFBg//TAMHBQ8YJBgg+fX6/+z+BwgMKtYmBSEGHg7+Evr+HgEV//kL//0D9fwAEAgE+woDJPkF/tgz8wD/Iewq/QsH9QYKHfEoAyoFHRH9//wFDvsUCf76CQECFCcFF/oRDxL4DfnnAP4MD/0h+gL7+/oLCRbV9fPnFAIGCDr9EgMWAvpJ8f/2FTP8/QT7BA8P6BYLIRH76wsf/foVChXwDgMMDRsMBRzm+QX4+/UMDgXdBxQm5fvoDf0JBQ7tGBwSHBwHMPgPAQEUCQskBhf9Bgj5ABf5CPn7HQ7wLxwIDQUl8PQc9/f09A/5CAcAAPgGAisSFfYCBPYeAPEo+xAJBCAIGC0UEBsLDgb8F/sw8xD1CioGB/sWBf374iUTDwnrJO77DBvYC/El+engBBT9IeUlJhYP9PsFC+/uI+kO6/sR/wcjIPkS8AUZ/ggRBwgO4uv7BvUgAdgb/QIe9Qb1AxgAAhIT3RfYKPDk+AAo+gzeFxwUBuD++hLxCh7dCgUdAgYF/gQNFgsLIiITBRD1CQwfERcYD/wGFw8EG+0i9AcI+/8tFOkG4yr/6xT9CQ3sFQUR+9/wCfUN8Q0CA+31DxnyDw0HGQkQ/vv+GwYNDg0H//sRDhT2DA0SBA/pGdUkDx4BBx3tFgMr69YM8xoG8wUF/isQ/QgIFvz7DhDqFf4hDfASGyYHEewfD/ICGfL49RkjGBQT6fkc7vcfHgIU8RYT8RQV9gjnHgjf7QT1Bw/9CxP57fv0GP0LDgL0FSX4BAP3JQ0JBxgbABQvBhUHBwvwHhX1H+Me8y0FEw4aAff8EeoH/Rj/+hgQAAL18hb+DCAMAtUADfcT+vke9gf++wEhCSD1DQIgLhcOBCP48gAJBe8IAxP9Ahkf8yQjDPkCCxEEEPr/+/4RAPDm+uoMA/cfERXg6Az2AOz7CPIODwMRFhMqBfkC8A4GCyIR8Qn8DgQZGu82DhH+EQwVBxkBCAcMFAMABAAL7QER+PvyDwACCwgKtRPkQBILMi4t1QvtHCziJh0jHBAA+Q8H7Bv3EfEpJToWCQUDDgX9Gx7uAPEbABYKCQQDCRwCABf8AgriIv0K8NAe+TP6MCsqJPLq7ALy5Bf1IPQBEAoBIfUYEAsCBg4WGxod5x4M/xUy5wn2FAXw+xYEBAH0AQDwzBsQCQX7FRs51xHjGt/TF/4l9hTxAhcqECX0CQIDDCL/9QYF/v4IKiIKF/kp/eIGFhfxFAcQCgAAAAAAAAAA+9YJ9//4GAgZ/OgJ4hXm8RYcBPsB7/4BBP8JDwn5EBINB/IrARYe+CQWD/z+AuwA+gsg9wsMHA8AAAAAAAAAAODiIPUP6CoPLtgOBw8WEQI1Huok7wYAABIM/QH2BR0VEgAWEgYUHBMPLAQNACYSAhUvDhT6FRfgHhLxC/Yh+/MC9hD8/fwhCx4AFgUCAPMNKM8G/xUY9w79GgIQACTz+iEHAAsUEP4GEgX+AAwFA/jq3xzzH94hCCoT7QLuKAH08+sN7Qb8/CAG0/r5FgwCExHuAxURIP0SKifzCBEzAvobEQoH9uvqKScOAPf9IBQCHBsGHf4SDx0AAAAAAAAAAPHO/QMcERYKE8n7AvgX+AkbC/wHEwEVAT0SBPMHGiMSMR/1J/MQ9/gPGv0DDgEL/h0Z6gQOBBYIF/YEIAUC9ej/AuYT9Q0cBx3c9ggr+Q/7DvL38xAWLxMUAw37CfYAFBMsAwMDJBAOHTALCREyDPIRDg38DvQXBQAAAAAAAAAA5twIEwn6+hgwCfAc6QPf8gMMChf6+i0EHgMXHewdG/RB6xzZCyAk9A0MHPwnFxQh6OIp8RfrFC0AAAAAAAAAAPsAGgof3hk0MBXJEtgu49AdFOD+2RTk7g0BJQQ+5RsOJfks+h7fFA38Bh8JEe0gCA8SIAcJ6woPAAAAAAAAAAAiFfX/5Akc+iIp7C7dEs4GEPUN/Az4FwIfCfsTAgIIBQwMCP4CHx3uICMNHAshJhra+fX4CvwJEAAAAAAAAAAACOPgEfgI9+8g/+kbCAMa6C0ABhIOFQr9CQkw/BX9GRIdCAH7+xwJBx01CSHtIPvwGhgL8SL1EAEAAAAAAAAAAAjVHQMG/Rn2GBcAB/sCFwUR/v8BFRMgEjUt8AECGQ77CPMLFfYd9/shJvcJHhUS8+jNI9Qx8CcWDBXzCQEg+toNIOn17jfX/Rb/+Q4RFBkNGBn8GuYw9/kpGhUGHBwdBiENBxUNCyEUBN4O8gsFGRcN+iL2OPUbLQwH+gMAFPjtGhvlGfULEAcdCvcG/Bn49QoRIgsk/RcQFwIf8yoJJgsBCPYX6/IB5C0O9wjkIg0ODf78GwEHAv8AAAAAAAAAAPzyEAn8EQjcEf8UEgcDHw4e/QQQDgkUHiEiCgULKf3uF/EGDQ7h8wo5Gfvu7RH/5R4c9xwCFQXtAAAAAAAAAADv6B/7IvYdEwb/9gcIH/keIPL//wMgJvn+IwQKJPELHCwCBAP3Cg/8GRsDDfUf/uIO/v4E/xD3/AAAAAAAAAAADuw3HB73FA0yFeMB3S0L9x8TEe8DAw8RIQ4fBPQRBPUd+xIDCfkiCyD0/QEN+xgd+v0P/w/2EwwAAAAAAAAAAPj/FwgAGQIOEAsJBBEODQkYJf8M9yj+8CMDFRYu9hQlFQ7+9wbzGe/x7hj8CgsE7gn9FgX/9woACQH6B+8R7eLqBRrzJxQZ3iIR2wfrQQDpCwEK/RoCGBIRExIVHysODxY++iEEEfnPGyQECAQLDfkaDQADBA0CBAAAAAAAAAAABBgG9wUfAQsPCQUF9BcMAhIEHwosGyLzNB7y8hjnHApKKAML7ywN/gwKJwIQ/hX3BCIG+iQMEgwAAAAAAAAAAB/98g/5HvnuIiUN/AcTFAA2ER0gABf9EjUFJ/gL/ikgB/r+Gv/nE/0UDPj6HPca8A8WBOoPERv5Fgf8BgMg+ewP/fgT9hMeAh4K/hD5ER4RC/8UFQoYEiEgGRr0Bw8J7ycV+TYIBPDv+fwc8x/hHgcF6D3tG/ErLAAAAAAAAAAADekf/xwF+w4PHQEH//oH/Qfm/xMt9g0j8wAj9hASDPweIvsnBf778CEEEBLsIPcAKP4AAv0R4/oAAAAAAAAAAPwFDQ/zCA4DExkQGOoXE+oi9Cn9EBwf+zsRBh0ILBEIAQ4wHiXoBB0oJtcZ3R365Uf9Ag0AHPkJ9+kZ7xnoHB735/b+F+UZHQrwDxgR/x8XGyMKGuYp/v4gGwPyAwIBFBYDEgoWCyQVM/4D9vQT+QgR6e4A3Qzo3QAAAAAAAAAAFxj4/egH4eII7hsHFQwJHRINEQ7+KAPyIRYGFQgTCgk9AQftJgkfKfb+8gEDBf78DgDSEOgV6eoAAAAAAAAAAAbd/fgcCAwSFPIoEi7lDxkQBhAo8P8R6hoa+RYMFRnuIzsNGOsu8ewmB/YF+SP89QwC5RD4CvHaBADlFu8T+v4GxyoAB+wmKxsT8vz2ERPvG/AU5zf4ByUhERoFDyUaJTTkHPIWCg4VMAfiAwML9usOAPIF7xvy+BAB+P/uGP8CChf+HNkOEewTQvL44iQHAwbn9vsMDyIEJiX89xIO8BICBiPn/voOBx8IBTMHFxoQDAn0BfMs8vUCAPEG7hH47g8C3RUK/PfeDiIO7hYV8Q8YBhYHIg8VFgzpF/ARIiT0BSEfJvge6vkcCQoKEgUP6RDqJeou3x8cEwDtAe0X89sR7Pn/DgEa9BwP//cWIwgHFQkm4C4DEwkcIvcGMRYjDwIYCCgABwgS+w8RFe4DHwAD+CT3HwMgCwAAAAAAAAAAHO0UFuwDFvALHN0GByEGC/8MBu0dDBYTPesc4xv6Ehf8/hsjGeT6DC4j1BDYDdXOB/7vCPMO5gTw6RnuGuoXDAhO7AnpCQX9EyMcDgMXCvMMCyLz5BEW9wvXFeol/jAIHCkIB/j95xoRABkQBAclASIL8Rf2D/vf6+gi7xnpGBMHB/Mf+wf6AhYFHPsDAQsEFgUICw4eHREgDwgP+vv5CxsACP4LDfkFHu80DSb1DhEN6d34//7p3AAAAAAAAAAAAjL7CwkPDvwqGhAN9gAeHBcYDPX9GPz/DgwGBB0EHAQI9BrjB/cgJBsTEBUR5QT3F+jw/eQN8eYAAAAAAAAAABYBDAwbFBT+FC/qDNspGOslDQUCDwET+AjwHAAZBB/5MQEPAxQgEfUH7gj9HfgjD/Dt+fcS3AMDAAAAAAAAAAAdGM4NFv8O9Sso7/XuBv7vEh8LAyYCBBI2HfQGBQv66kMK3goSHwsS/RkAARkD/gbz9+z8Dv0Y/wUA/QbuFfrvD8IfFh4IDiwOD+X3/BwD9BzrBvgrEyAOJAcHDfoOCBUgFBfsFP//DBj9CxcEEQbh8vYF/fX6Av8MAOz37hn0DyH07wHwJCDnGBYICCsBEAwDGwH48gID+ygL9PgtMj8RHg4CFQAD6QHy3xEZLf0qAAfqJP5E6S8jBwDwA+8V/vMa8PIM+wsHBzEk6AgKEA0XFf4J+yUaGScf/hvzCwQQES71AwLz+gkGIBHdG+sj2usK9xL+Ed4N/wEA4wDtC+rqGAUdAg0GHgQOCRXfHvoxEi0P//z2Axb4LuoNECMGGQglEPwP7PbnBvgBC/kIEBHzF/fu6gIJ3Onv6RnvGegOExsPCAzzCQ/0IfgV3hzpEg8oRuMZ+RwO+xjjGAUW+hIqEyHo8w/9+fscHfYP/igG/Rf5BOwfAe8O8ugd7hvhKh4TDgEc/RwIBRQfBgAQ/grhKQkKBQr8EfsRCwsM2zMF7A/tNfcm6TAfJhPxAvEA+Qj5+h38KwUHC/bqIesc5CQaDwb6JfgB/AAn+SENLQogCyksDvv9AhICLirfA/cQ9e0NBu/7AhQa8gj39gj7FhYH7sYg9DLmCijc5h3sGuYlIQcM6Rr5Fv/xJzP+CPkeDtgEBhYCGggTAwMBBg8mCxIVECfz9+j3+gX3/QL3GPMgFvT9Fvk3/SIJ8gAN/QbvCw4TAfQM8f8J2A8XDOsn+RMHIin+9PQgAOgl9wYXDgMP/xz7BgsEBQYdGOv87AfmBwENCA8FA/n/BPLzGfkI5xEACf4IFwz/CBcC9AAEJQMqAxQLN/I1DTAnKvwI/yQeGAAmBPAS7uwEFCriHOcN/wEAFB4C8/ISDw3u6g0IE/AJDCMm6wnvGxH6Ew8RAwYNDBkLGvf9AwMFHQ8FBv0GCRQNB/7s3/wIARYaMfobAxAM6yAG9vT7Hezz+PYCBA3+GQAUAu0T7gESCQYt6RvmGBjyFg0VAAwGGA0fIP0S+TIJ+QYF+OwgA/AB8PceGAwAGvn1+BPcIvYDGwoA+AXvEPfuEv/6GAMUBfUU/Rr9JgUlFQ4iARf+Fgv7BQ0RBQAdKA4U9Ab6EA/7EAAI5gEJCCrwBscP7BnsAhfs6hTvINkXHyAW4vkGBxf/CwQe9BnVHhEtKxAFFBQXEiED8O8HEgwNFwjz+fklIQLzGv8t/vn7/wvdDfYF8TgbEb8U5AXv9wUOCgkMCxH0AAskCwXiGBD8ERYYBRcHGhUsBfEo/xMbEAvwEgoiEQsQ+RwRCtgUDQDmBS/nJ9ohH/vODd0X7xb4AhXsCwgY7PYjFRvwRMswGAkd7/IPFQMIGPITJwQOAfwcH/sLCCcEEAoQHxQc6QgRDgYAD/8X8wf7uRjfEPcRKhUM8grgBAXuHRkdAB35GBomGBD5FwwUEBblFQgMECgYHRPr/god2gDqBfAc+hP7BvEDEQQCAv8K5cUnBAr4IyIZFNIB8wcB7/f3Fw4k7x8bDx7yEwwQEPoW9wMC9AoFLREDBQIMAAL+CBDxAPUO/Pfw2xrqKPUbMgUg+xn2IwIGDfbmCun6/d0fDfb5/gADAhcvDQz4EhH2GS/z9/kWAP8O7yAII+4ZOgoJGPX7DN71EPoO+AIH9/UQNgLyAQ0YBCTZFisT8w0JIhMXBwsAIQogDfIT4RgH6xv78P8TExr8IO0S3B8MCwsVMewa6Rzc8+vhLO8H2RsvzMQ29DjVFy0B9wUPDx4NAPf4F+n9ASoYGg4c7wAVEPkQLA/w+Az9CAwVBAML9fvwGxkN5v4ICBQWAwQY5A3j5Pz0/gX9/P4JGwDtEfMWA+8PBTYAGesdEP8iBwkZFQQECfgUFf0KEQo3Bx0HDgkG5fAL4CkLDPQSCuUf8RwUJRkD5CLkDAMJLCf66yfxFOr5D90o20XmEA4WEwgIDBYEBh4ECf4RJvj5CADqEewW7Rj2FwgOEvQBDRfZJt0XCvEW3uoi2iHeISIQDAwG2CIDEwER//oZ4ykWHgUWCzEGBQgACgAQCRse9S/39fkFCyj9/Q0lLOYF5voGFQr3D/weFRG4E+0k8OoBFff2/uz4A/wLCyIUE/EZ+R37EPgLGR8DHfn3IQkEJPwl6g4FChv1Bgcs/hbnDfQFC/wA+g3xLQMB1RL8/gYQBR31/CbuCwPk9flI6yj2CiMXJCwCCw8U9AwF+AkSFBUNLfXyBCQZEyXr7w/7D94CAeclAQf3BgD8+BgVEhr5PAgZEdoHyhr6x/0qG/Yv7C4pDfwOCSMKD/sV/wgH9uccDBkP7gv/C/HtCA4NFRgIHQP7BwTxCxP5AAHW/Qf+CQfxCwDm8PYTA+kGEygEHNwe9g0W+PHyDgTgGQUSGxUtB/8n8v7l+ScH/vYSBxAP7fsF2cgR7iviNh3syh0GKsIWNQjvBP38DxXyCwcZAhcCAAAP/hcKHggQGxYNAgES+QP79QYSBPz0/QIv8P7yFxz9EAgd5Rb+IP3vCegEGAb+DQkFGgsg9hvy7yMeHAYVFR4VICcLAvEgAAsrHQz9+Q0j/Sb0/tkFAwHwBiQDAgcJ3xTnuQcBGPocGffEHQwq5hc6Bf0CBg0UEwwOAdguzyjf1CMC+Bj2BQEOEugYAwMFIx4LHwruKCYF7f8WEQf3KQ3aGxYECf0JBPI0//kLCQb7FvrwDPUa7Q8NEdgmBSX1LCkY/xELDwMh5CL+CgkGCQkbCucDBQH8+/8YAAf9JAr1DxYE/hH//vb94xH9+hAHJgsX98wHxxPvvB0CzTDbQfzPEs4NDiEBFwsaIAwHDQEqAxMKEw8JAAkCFRvl7fUg9BQh/wUXCx72CgPnDusl/RjM+AD69R/PGBEPEh0YJBICFQYMGgULEBjyJOz38v0F8OsE3B0LCwL2+Poa+g3+8N4R/xgB4v4NFwANAM8k/jfQ4hMHAe/mCwvUEgG4DcYq2uIV6v/y+xjqLBrxGh8fEBMUGxAmBfoHDPoRGvIMEQ8A7CkH5wIVF/D6Jwzb+wQlE/AU3xvqEN0MIADsFfEm+hUqJx4ABhYJEAoPJgX6FBf16hcFFvIPCAns/gYcF/rlGCMZKRf0/gH1DPsNFPwL+gYQKwPdH8wVBMd/I9Ye/scTykIIHu6iRQ7GO/PROTwC1fQVxese2Q3RCB4QBxo8+uDdKgc/mQUDzxEquM0LLCYZzc35z1RkD+z+MfLlh4Hm22o=";

}  // namespace papersoccer::turn_action_v2::replay_value_model

namespace papersoccer::turn_action_v2::teacher_residual_model {

inline constexpr std::size_t kInputCount = 24;
inline constexpr std::array<float, kInputCount> kWeights{{-0.000212487676F, -0.0314205534F, 0.00000000F, 0.0220964056F, 0.00318384444F, -0.0459056867F, 0.00000000F, 0.00000000F, -0.000812118322F, -0.0889680974F, -0.0757096574F, 0.00821899013F, -0.163250396F, -0.206006910F, 0.0891054922F, 0.00699152156F, -0.000185926722F, 0.0146061040F, 0.127743796F, 0.0246996844F, 0.00000000F, 0.00000000F, 0.00000000F, 0.00474756504F}};
inline constexpr float kBias = -0.0212487676F;

}  // namespace papersoccer::turn_action_v2::teacher_residual_model

namespace papersoccer::turn_action_v2 {

constexpr std::uint32_t kFirstSearchTimeMs = 800;
constexpr std::uint32_t kLaterSearchTimeMs = 165;
constexpr std::uint32_t kMaximumTurnDepth = 32;
constexpr std::uint64_t kMaximumNodes = 3'000'000;
constexpr std::size_t kTranspositionEntries = 262'144;
constexpr std::size_t kEvaluationEntries = 131'072;

constexpr int kMateScore = 1'000'000;
constexpr int kInfinity = 2'000'000;
constexpr int kMaximumEvaluation = 100'000;
constexpr int kDirectGoalWeight = 50'000;
constexpr int kGoalDistanceWeight = 120;
constexpr int kVerticalProgressWeight = 80;
constexpr int kContinuationWeight = 35;
constexpr int kMobilityWeight = 20;
constexpr int kForwardMoveWeight = 10;
constexpr int kCenterAlignmentWeight = 6;
constexpr int kTempoWeight = 15;
constexpr int kTeacherResidualTargetScale = 20'000;
constexpr int kTeacherResidualHardCap = 6'000;
constexpr int kTeacherResidualMinimumUsedEdges = 12;
constexpr std::size_t kTeacherResidualInputs = 24;

constexpr std::array<Point, 8> kDirectionDeltas{{
{0, -1},
{1, -1},
{1, 0},
{1, 1},
{0, 1},
{-1, 1},
{-1, 0},
{-1, -1},
}};

class SearchBudgetReached final {};

using SearchClock = std::chrono::steady_clock;

enum class ScoreBound { Exact, Lower, Upper };
enum class ReboundOutcome { Unknown, Win, Loss };

constexpr std::uint8_t kExactProofRootGoal = 1U << 0U;
constexpr std::uint8_t kExactProofLeafBoundary = 1U << 1U;
constexpr std::uint8_t kExactProofPlyOne = 1U << 2U;
constexpr std::uint8_t kExactProofPlyTwo = 1U << 3U;
constexpr std::uint8_t kAllExactProofs =
kExactProofRootGoal | kExactProofLeafBoundary | kExactProofPlyOne |
kExactProofPlyTwo;

struct SearchConfig {
std::uint32_t max_turn_depth{kMaximumTurnDepth};
std::uint64_t max_nodes{kMaximumNodes};
std::size_t transposition_entries{kTranspositionEntries};
std::size_t evaluation_entries{kEvaluationEntries};
std::uint32_t max_time_ms{};
std::optional<SearchClock::time_point> absolute_deadline{};
bool root_seed_endpoints{true};
bool terminal_bound_pruning{true};
bool root_transposition_pruning{true};
std::uint8_t exact_proof_mask{};
int replay_value_blend_percent{};
int teacher_residual_weight_percent{};
};

struct SearchStats {
std::uint32_t completed_turn_depth{};
std::uint32_t attempted_turn_depth{};
std::uint64_t nodes{};
std::uint64_t leaf_evaluations{};
std::uint64_t terminal_nodes{};
std::uint64_t completed_actions{};
std::uint64_t cutoffs{};
std::uint64_t transposition_probes{};
std::uint64_t transposition_hits{};
std::uint64_t transposition_cutoffs{};
std::uint64_t transposition_stores{};
std::uint64_t continuation_transposition_hits{};
std::uint64_t evaluation_cache_probes{};
std::uint64_t evaluation_cache_hits{};
std::uint64_t teacher_residual_evaluations{};
std::uint64_t terminal_bound_cutoffs{};
std::uint64_t rebound_goal_probes{};
std::uint64_t rebound_goal_hits{};
std::uint64_t rebound_loss_hits{};
std::uint64_t root_rebound_probes{};
std::uint64_t root_rebound_win_hits{};
std::uint64_t root_rebound_loss_hits{};
std::uint64_t leaf_rebound_probes{};
std::uint64_t leaf_rebound_win_hits{};
std::uint64_t leaf_rebound_loss_hits{};
std::uint64_t exchange_ply1_probes{};
std::uint64_t exchange_ply1_win_hits{};
std::uint64_t exchange_ply1_loss_hits{};
std::uint64_t exchange_ply1_cutoffs{};
std::uint64_t exchange_ply2_probes{};
std::uint64_t exchange_ply2_win_hits{};
std::uint64_t exchange_ply2_loss_hits{};
std::uint64_t exchange_ply2_cutoffs{};
std::uint64_t forced_edges{};
std::uint64_t root_seed_actions{};
std::uint64_t root_transposition_reuses{};
std::uint32_t max_action_edges{};
int root_score{};
bool budget_exhausted{};
};

struct TranspositionEntry {
detail::PositionKey key{};
std::uint32_t depth{};
int score{};
Move best_move{};
ScoreBound bound{ScoreBound::Exact};
bool occupied{};
};

class TranspositionTable {
public:
explicit TranspositionTable(std::size_t entries) : entries_(entries) {}

const TranspositionEntry *find(detail::PositionKey key) const noexcept {
if (entries_.size() < 2) {
return nullptr;
}
const std::size_t bucket = bucket_index(key);
for (std::size_t way = 0; way < 2; ++way) {
const TranspositionEntry &entry = entries_[bucket + way];
if (entry.occupied && entry.key == key) {
return &entry;
}
}
return nullptr;
}

bool store(detail::PositionKey key, std::uint32_t depth, int score,
Move best_move, ScoreBound bound) noexcept {
if (entries_.size() < 2) {
return false;
}
const std::size_t bucket = bucket_index(key);
TranspositionEntry *victim = nullptr;
for (std::size_t way = 0; way < 2; ++way) {
TranspositionEntry &entry = entries_[bucket + way];
if (entry.occupied && entry.key == key) {
if (entry.depth > depth) {
return false;
}
victim = &entry;
break;
}
if (!entry.occupied || victim == nullptr || entry.depth < victim->depth) {
victim = &entry;
}
}
if (victim == nullptr || (victim->occupied && victim->depth > depth)) {
return false;
}
*victim = TranspositionEntry{key, depth, score, best_move, bound, true};
return true;
}

private:
std::vector<TranspositionEntry> entries_;

std::size_t bucket_index(detail::PositionKey key) const noexcept {
const std::uint64_t combined =
key.first ^ ((key.second << 23U) | (key.second >> 41U));
const std::size_t buckets = entries_.size() / 2U;
return 2U * static_cast<std::size_t>(combined % buckets);
}
};

struct EvaluationEntry {
detail::PositionKey key{};
int score{};
bool occupied{};
};

class EvaluationTable {
public:
explicit EvaluationTable(std::size_t entries) : entries_(entries) {}

std::optional<int> find(detail::PositionKey key) const noexcept {
if (entries_.size() < 2) {
return std::nullopt;
}
const std::size_t bucket = bucket_index(key);
for (std::size_t way = 0; way < 2; ++way) {
const EvaluationEntry &entry = entries_[bucket + way];
if (entry.occupied && entry.key == key) {
return entry.score;
}
}
return std::nullopt;
}

void store(detail::PositionKey key, int score) noexcept {
if (entries_.size() < 2) {
return;
}
const std::size_t bucket = bucket_index(key);
for (std::size_t way = 0; way < 2; ++way) {
EvaluationEntry &entry = entries_[bucket + way];
if (!entry.occupied || entry.key == key) {
entry = EvaluationEntry{key, score, true};
return;
}
}
const std::uint64_t combined = combined_key(key);
entries_[bucket + ((combined >> 32U) & 1U)] =
EvaluationEntry{key, score, true};
}

private:
std::vector<EvaluationEntry> entries_;

std::uint64_t combined_key(detail::PositionKey key) const noexcept {
return key.first ^ ((key.second << 29U) | (key.second >> 35U));
}

std::size_t bucket_index(detail::PositionKey key) const noexcept {
const std::size_t buckets = entries_.size() / 2U;
return 2U * static_cast<std::size_t>(combined_key(key) % buckets);
}
};

class ScopedMove {
public:
ScopedMove(detail::SearchPosition &position, std::uint8_t slot)
: position_(position) {
position_.make_move(slot);
}

~ScopedMove() { position_.unmake_move(); }

ScopedMove(const ScopedMove &) = delete;
ScopedMove &operator=(const ScopedMove &) = delete;

private:
detail::SearchPosition &position_;
};

struct OrderedMove {
std::uint8_t slot{};
Move move{};
int score{};
};

struct OrderedMoveList {
std::array<OrderedMove, detail::kMaximumMoves> values{};
std::uint8_t count{};
};

struct ActionSearchResult {
int score{};
Move first_move{};
};

struct RootResult {
int score{};
std::vector<Move> action;
};

struct EvaluationSnapshot {
int anchor_score{};
int mover_sign{};
std::array<float, kTeacherResidualInputs> features{};
};

struct ReplayValueOutput {
float logit{};
std::array<float, replay_value_model::kHiddenTwo> hidden{};
std::uint16_t used_edges{};
};

int player_sign(Player player) noexcept {
return player == Player::One ? 1 : -1;
}

class CompleteTurnSearch {
public:
CompleteTurnSearch(const GameState &state, SearchConfig config)
: config_(config),
topology_(std::make_shared<detail::SearchTopology>(state.config)),
position_(topology_, state),
table_(config.transposition_entries),
evaluations_(config.evaluation_entries),
distances_(topology_->vertex_count(), -1),
queue_(topology_->vertex_count()),
rebound_seen_(config.exact_proof_mask != 0 ? topology_->vertex_count()
: 0),
teacher_residual_root_enabled_(
state.used_segments.size() >=
static_cast<std::size_t>(kTeacherResidualMinimumUsedEdges)) {
if (config_.max_turn_depth == 0 ||
config_.max_turn_depth > kMaximumTurnDepth || config_.max_nodes == 0) {
throw std::invalid_argument("invalid complete-turn search configuration");
}
if (config_.exact_proof_mask > kAllExactProofs) {
throw std::invalid_argument("invalid exact proof mask");
}
if (config_.replay_value_blend_percent < 0 ||
config_.replay_value_blend_percent > 100 ||
config_.teacher_residual_weight_percent < 0 ||
config_.teacher_residual_weight_percent > 100) {
throw std::invalid_argument("invalid value evaluation configuration");
}
initialize_rotation_maps();
if (config_.absolute_deadline.has_value()) {
deadline_ = config_.absolute_deadline;
} else if (config_.max_time_ms != 0) {
deadline_ =
SearchClock::now() + std::chrono::milliseconds(config_.max_time_ms);
}
}

std::vector<Move> run() {
if (position_.is_terminal()) {
throw std::invalid_argument("cannot search a terminal state");
}

if (exact_proof_enabled(kExactProofRootGoal)) {
std::vector<Move> rebound_goal;
++stats_.root_rebound_probes;
const ReboundOutcome rebound = analyze_rebound_component(&rebound_goal);
if (rebound == ReboundOutcome::Win) {
++stats_.root_rebound_win_hits;
stats_.completed_actions = 1;
stats_.max_action_edges =
static_cast<std::uint32_t>(rebound_goal.size());
stats_.root_score = immediate_win_score(position_.to_move(), 0);
return rebound_goal;
}
if (rebound == ReboundOutcome::Loss) {
++stats_.root_rebound_loss_hits;
}
}

std::vector<Move> committed_action = fallback_action();
for (std::uint32_t depth = 1; depth <= config_.max_turn_depth; ++depth) {
stats_.attempted_turn_depth = depth;
try {
RootResult result = search_root(depth);
committed_action = std::move(result.action);
stats_.completed_turn_depth = depth;
stats_.root_score = result.score;
if (std::abs(stats_.root_score) >=
kMateScore - static_cast<int>(kMaximumTurnDepth)) {
break;
}
} catch (const SearchBudgetReached &) {
stats_.budget_exhausted = true;
if (stats_.completed_turn_depth == 0 && captured_any_ &&
!captured_action_.empty()) {
committed_action = captured_action_;
stats_.root_score = captured_score_;
}
break;
}
}
return committed_action;
}

const SearchStats &stats() const noexcept { return stats_; }

EvaluationSnapshot evaluation_snapshot() {
return make_evaluation_snapshot();
}

#ifdef PAPER_SOCCER_HYBRID_EXACT_PROOF_TESTING
RootResult run_fixed_depth_for_test(std::uint32_t depth) {
stats_.attempted_turn_depth = depth;
RootResult result = search_root(depth);
stats_.completed_turn_depth = depth;
stats_.root_score = result.score;
return result;
}
#endif

private:
SearchConfig config_;
std::shared_ptr<const detail::SearchTopology> topology_;
detail::SearchPosition position_;
TranspositionTable table_;
EvaluationTable evaluations_;
SearchStats stats_;
std::vector<int> distances_;
std::vector<detail::SearchTopology::VertexIndex> queue_;
std::vector<std::uint32_t> rebound_seen_;
std::uint32_t rebound_generation_{};
std::deque<detail::SearchTopology::VertexIndex> distance_queue_;
std::vector<detail::SearchTopology::VertexIndex> rotated_vertices_;
std::vector<detail::SearchTopology::EdgeIndex> rotated_edges_;
std::optional<SearchClock::time_point> deadline_;
std::vector<Move> current_action_;
std::vector<Move> captured_action_;
int captured_score_{};
bool captured_any_{};
bool teacher_residual_root_enabled_{};

bool exact_proof_enabled(std::uint8_t proof) const noexcept {
return (config_.exact_proof_mask & proof) != 0;
}

void visit_node() {
if (stats_.nodes >= config_.max_nodes) {
throw SearchBudgetReached{};
}
if ((stats_.nodes & 15U) == 0U && deadline_.has_value() &&
SearchClock::now() >= *deadline_) {
throw SearchBudgetReached{};
}
++stats_.nodes;
}

detail::PositionKey boundary_key() const noexcept {
detail::PositionKey key = position_.position_key();
detail::xor_position_key(
key, detail::position_key_component(6, position_.ball_vertex()));
return key;
}

int terminal_score(std::uint32_t turn_ply) {
++stats_.terminal_nodes;
const std::optional<Player> winning_player = position_.winner();
if (!winning_player.has_value()) {
throw std::logic_error("terminal state has no winner");
}
const int distance = static_cast<int>(
std::min<std::uint32_t>(turn_ply, kMaximumTurnDepth));
return *winning_player == Player::One ? kMateScore - distance
: -kMateScore + distance;
}

int immediate_win_score(Player mover, std::uint32_t turn_ply) const {
const int distance = static_cast<int>(
std::min<std::uint32_t>(turn_ply + 1U, kMaximumTurnDepth));
return mover == Player::One ? kMateScore - distance
: -kMateScore + distance;
}

bool reached_immediate_win_bound(int score, Player mover,
std::uint32_t turn_ply) const {
return config_.terminal_bound_pruning &&
score == immediate_win_score(mover, turn_ply);
}

Point rotated_point(Point point) const noexcept {
return {topology_->config().width - point.x,
topology_->config().height + 2 - point.y};
}

void initialize_rotation_maps() {
using VertexIndex = detail::SearchTopology::VertexIndex;
using EdgeIndex = detail::SearchTopology::EdgeIndex;
constexpr EdgeIndex kMissingEdge =
std::numeric_limits<EdgeIndex>::max();

rotated_vertices_.resize(topology_->vertex_count());
for (std::size_t vertex = 0; vertex < topology_->vertex_count(); ++vertex) {
const auto rotated =
topology_->find_vertex(rotated_point(topology_->point(vertex)));
if (!rotated.has_value()) {
throw std::logic_error("rotated vertex is missing from topology");
}
rotated_vertices_[vertex] = *rotated;
}

rotated_edges_.assign(topology_->edge_count(), kMissingEdge);
for (std::size_t source = 0; source < topology_->vertex_count(); ++source) {
for (const Player player : {Player::One, Player::Two}) {
const auto &adjacency = topology_->adjacency(
static_cast<VertexIndex>(source), player);
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
const Segment rotated_segment{
rotated_point(topology_->point(source)),
rotated_point(topology_->point(arc.destination))};
const auto rotated_edge = topology_->find_edge(rotated_segment);
if (!rotated_edge.has_value()) {
throw std::logic_error("rotated edge is missing from topology");
}
rotated_edges_[arc.edge] = *rotated_edge;
}
}
}
if (std::find(rotated_edges_.begin(), rotated_edges_.end(), kMissingEdge) !=
rotated_edges_.end()) {
throw std::logic_error("rotation map does not cover every edge");
}
}

int shortest_goal_distance(Player player) {
std::fill(distances_.begin(), distances_.end(), -1);
std::size_t head = 0;
std::size_t tail = 0;
const auto start = position_.ball_vertex();
distances_[start] = 0;
queue_[tail++] = start;

while (head < tail) {
const auto vertex = queue_[head++];
const auto &adjacency = topology_->adjacency(vertex, player);
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (position_.edge_used(arc.edge) ||
distances_[arc.destination] >= 0) {
continue;
}
const int distance = distances_[vertex] + 1;
distances_[arc.destination] = distance;
if (is_attacking_goal(topology_->config(),
topology_->point(arc.destination), player)) {
return distance;
}
queue_[tail++] = arc.destination;
}
}
return static_cast<int>(topology_->vertex_count()) + 8;
}

ReboundOutcome analyze_rebound_component(std::vector<Move> *action) {
++stats_.rebound_goal_probes;
if (action != nullptr) {
action->clear();
}
if (++rebound_generation_ == 0) {
std::fill(rebound_seen_.begin(), rebound_seen_.end(), 0);
rebound_generation_ = 1;
}
const auto start = position_.ball_vertex();
rebound_seen_[start] = rebound_generation_;
distances_[start] = static_cast<int>(start);
std::size_t head = 0;
std::size_t tail = 0;
queue_[tail++] = start;
const Player mover = position_.to_move();
const RulesConfig &rules = topology_->config();
bool safe_handoff = false;

while (head < tail) {
const auto vertex = queue_[head++];
const auto &adjacency = topology_->adjacency(vertex, mover);
std::array<std::uint8_t, detail::kMaximumMoves> order{};
std::array<std::uint8_t, detail::kMaximumMoves> keys{};
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
order[index] = index;
const Point destination = topology_->point(adjacency.arcs[index].destination);
const Point source = topology_->point(vertex);
int dx = destination.x - source.x;
int dy = destination.y - source.y;
if (mover == Player::Two) {
dx = -dx;
dy = -dy;
}
for (std::size_t direction = 0; direction < kDirectionDeltas.size();
++direction) {
if (kDirectionDeltas[direction].x == dx &&
kDirectionDeltas[direction].y == dy) {
keys[index] = static_cast<std::uint8_t>(direction);
break;
}
}
}
for (std::uint8_t index = 1; index < adjacency.count; ++index) {
const std::uint8_t value = order[index];
std::uint8_t insertion = index;
while (insertion > 0 && keys[value] < keys[order[insertion - 1U]]) {
order[insertion] = order[insertion - 1U];
--insertion;
}
order[insertion] = value;
}
for (std::uint8_t ranked = 0; ranked < adjacency.count; ++ranked) {
const auto &arc = adjacency.arcs[order[ranked]];
if (position_.edge_used(arc.edge)) {
continue;
}
const Point destination = topology_->point(arc.destination);
if (is_attacking_goal(rules, destination, mover)) {
++stats_.rebound_goal_hits;
if (action != nullptr) {
distances_[arc.destination] = static_cast<int>(vertex);
for (auto current = arc.destination; current != start;
current = static_cast<detail::SearchTopology::VertexIndex>(
distances_[current])) {
action->push_back(Move{topology_->point(current)});
}
std::reverse(action->begin(), action->end());
}
return ReboundOutcome::Win;
}
if (is_goal_point(rules, destination) ||
rebound_seen_[arc.destination] == rebound_generation_) {
continue;
}
if (!is_boundary_point(rules, destination) &&
!position_.vertex_visited(arc.destination)) {
const auto &handoff =
topology_->adjacency(arc.destination, opponent(mover));
for (std::uint8_t child = 0; child < handoff.count; ++child) {
const auto edge = handoff.arcs[child].edge;
if (edge != arc.edge && !position_.edge_used(edge)) {
safe_handoff = true;
break;
}
}
continue;
}
rebound_seen_[arc.destination] = rebound_generation_;
distances_[arc.destination] = static_cast<int>(vertex);
queue_[tail++] = arc.destination;
}
}
if (!safe_handoff && rules.blocked_rule == BlockedRule::MoverLoses) {
++stats_.rebound_loss_hits;
return ReboundOutcome::Loss;
}
return ReboundOutcome::Unknown;
}

void calculate_goal_turn_distances(Player player) {
constexpr int kUnreachable = 1'000'000;
std::fill(distances_.begin(), distances_.end(), kUnreachable);
distance_queue_.clear();
const auto start = position_.ball_vertex();
distances_[start] = 0;
distance_queue_.push_front(start);

while (!distance_queue_.empty()) {
const auto vertex = distance_queue_.front();
distance_queue_.pop_front();
const auto &adjacency = topology_->adjacency(vertex, player);
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (position_.edge_used(arc.edge)) {
continue;
}
const Point destination = topology_->point(arc.destination);
const bool continues_turn =
is_boundary_point(topology_->config(), destination) ||
position_.vertex_visited(arc.destination) ||
is_goal_point(topology_->config(), destination);
const int edge_cost = continues_turn ? 0 : 1;
const int distance = distances_[vertex] + edge_cost;
if (distance >= distances_[arc.destination]) {
continue;
}
distances_[arc.destination] = distance;
if (edge_cost == 0) {
distance_queue_.push_front(arc.destination);
} else {
distance_queue_.push_back(arc.destination);
}
}
}
}

static int base64_value(char character) noexcept {
if (character >= 'A' && character <= 'Z') {
return character - 'A';
}
if (character >= 'a' && character <= 'z') {
return character - 'a' + 26;
}
if (character >= '0' && character <= '9') {
return character - '0' + 52;
}
if (character == '+') {
return 62;
}
if (character == '/') {
return 63;
}
return -1;
}

static const std::vector<std::int8_t> &replay_value_weights() {
static const std::vector<std::int8_t> weights = [] {
std::vector<std::int8_t> decoded;
decoded.reserve(replay_value_model::kInputCount *
replay_value_model::kHiddenOne +
replay_value_model::kHiddenOne *
replay_value_model::kHiddenTwo +
replay_value_model::kHiddenTwo);
std::uint32_t buffer = 0;
int bits = 0;
for (const char character : replay_value_model::kEncodedWeights) {
if (character == '=') {
break;
}
const int value = base64_value(character);
if (value < 0) {
throw std::logic_error("invalid replay value model encoding");
}
buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
bits += 6;
if (bits >= 8) {
bits -= 8;
decoded.push_back(static_cast<std::int8_t>(
static_cast<std::uint8_t>((buffer >> bits) & 0xffU)));
}
}
const std::size_t expected =
replay_value_model::kInputCount * replay_value_model::kHiddenOne +
replay_value_model::kHiddenOne * replay_value_model::kHiddenTwo +
replay_value_model::kHiddenTwo;
if (decoded.size() != expected) {
throw std::logic_error("replay value model has an invalid size");
}
return decoded;
}();
return weights;
}

ReplayValueOutput replay_value_output(Player mover) {
if (topology_->edge_count() != replay_value_model::kEdgeCount ||
topology_->vertex_count() != replay_value_model::kVertexCount) {
throw std::logic_error("replay value model topology mismatch");
}
const auto &weights = replay_value_weights();
ReplayValueOutput output;
std::array<float, replay_value_model::kHiddenOne> hidden_one =
replay_value_model::kB1;

const auto accumulate_first_layer = [&](std::size_t input) {
const std::size_t offset = input * replay_value_model::kHiddenOne;
for (std::size_t hidden = 0;
hidden < replay_value_model::kHiddenOne; ++hidden) {
hidden_one[hidden] +=
static_cast<float>(weights[offset + hidden]) *
replay_value_model::kW1Scale;
}
};

for (std::size_t edge = 0; edge < topology_->edge_count(); ++edge) {
if (!position_.edge_used(
static_cast<detail::SearchTopology::EdgeIndex>(edge))) {
continue;
}
++output.used_edges;
const std::size_t normalized_edge =
mover == Player::One ? edge : rotated_edges_[edge];
accumulate_first_layer(normalized_edge);
}

calculate_goal_turn_distances(mover);
for (std::size_t vertex = 0; vertex < topology_->vertex_count(); ++vertex) {
const std::size_t physical_vertex =
mover == Player::One ? vertex : rotated_vertices_[vertex];
const int distance = std::clamp(distances_[physical_vertex], 0, 7);
accumulate_first_layer(replay_value_model::kEdgeCount + vertex * 8U +
static_cast<std::size_t>(distance));
}
for (float &value : hidden_one) {
value = std::max(value, 0.0F);
}

constexpr std::size_t kSecondLayerOffset =
replay_value_model::kInputCount * replay_value_model::kHiddenOne;
std::array<float, replay_value_model::kHiddenTwo> hidden_two =
replay_value_model::kB2;
for (std::size_t first = 0; first < replay_value_model::kHiddenOne;
++first) {
for (std::size_t second = 0; second < replay_value_model::kHiddenTwo;
++second) {
const std::size_t weight = kSecondLayerOffset +
first * replay_value_model::kHiddenTwo +
second;
hidden_two[second] += hidden_one[first] *
static_cast<float>(weights[weight]) *
replay_value_model::kW2Scale;
}
}
for (float &value : hidden_two) {
value = std::max(value, 0.0F);
}

constexpr std::size_t kOutputLayerOffset =
kSecondLayerOffset + replay_value_model::kHiddenOne *
replay_value_model::kHiddenTwo;
output.hidden = hidden_two;
output.logit = replay_value_model::kB3;
for (std::size_t hidden = 0; hidden < replay_value_model::kHiddenTwo;
++hidden) {
output.logit += hidden_two[hidden] *
static_cast<float>(
weights[kOutputLayerOffset + hidden]) *
replay_value_model::kW3Scale;
}
return output;
}

EvaluationSnapshot make_evaluation_snapshot() {
const Player mover = position_.to_move();
const int sign = player_sign(mover);
const Point ball = position_.ball();
const RulesConfig &rules = topology_->config();

std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = position_.legal_slots(slots);
int continuations = 0;
int forward_moves = 0;
bool direct_goal = false;
for (std::uint8_t index = 0; index < count; ++index) {
const std::uint8_t slot = slots[index];
const Point destination = position_.move_for_slot(slot).to;
direct_goal = direct_goal ||
is_attacking_goal(rules, destination, mover);
continuations += position_.grants_extra_turn(slot) ? 1 : 0;
const bool forward = mover == Player::One ? destination.y < ball.y
: destination.y > ball.y;
forward_moves += forward ? 1 : 0;
}

const int player_one_distance = shortest_goal_distance(Player::One);
const int player_two_distance = shortest_goal_distance(Player::Two);
const int distance_advantage = player_two_distance - player_one_distance;
const int vertical_progress = rules.height + 2 - 2 * ball.y;
const int center_alignment =
rules.width / 2 - std::abs(ball.x - rules.width / 2);

int hand_score = distance_advantage * kGoalDistanceWeight +
vertical_progress * kVerticalProgressWeight;
hand_score += sign * (static_cast<int>(count) * kMobilityWeight +
continuations * kContinuationWeight +
forward_moves * kForwardMoveWeight +
center_alignment * kCenterAlignmentWeight +
kTempoWeight);
if (direct_goal) {
hand_score += sign * kDirectGoalWeight;
}
const ReplayValueOutput replay = replay_value_output(mover);
const float replay_probability = std::tanh(replay.logit * 0.5F);
const int replay_score = sign * static_cast<int>(
std::lround(replay_probability *
kMaximumEvaluation));
int anchor_score = hand_score;
if (config_.replay_value_blend_percent != 0) {
anchor_score =
(hand_score * (100 - config_.replay_value_blend_percent) +
replay_score * config_.replay_value_blend_percent) /
100;
}

EvaluationSnapshot snapshot;
snapshot.anchor_score =
std::clamp(anchor_score, -kMaximumEvaluation, kMaximumEvaluation);
snapshot.mover_sign = sign;
for (std::size_t hidden = 0; hidden < replay.hidden.size(); ++hidden) {
snapshot.features[hidden] =
std::clamp(replay.hidden[hidden] * 0.25F, 0.0F, 1.0F);
}
snapshot.features[8] = replay_probability;
snapshot.features[9] =
static_cast<float>(sign * hand_score) / kMaximumEvaluation;
snapshot.features[10] =
static_cast<float>(sign * snapshot.anchor_score) /
kMaximumEvaluation;
const Point canonical_ball =
mover == Player::One ? ball : rotated_point(ball);
snapshot.features[11] =
static_cast<float>(canonical_ball.x - rules.width / 2) /
static_cast<float>(rules.width / 2);
snapshot.features[12] =
static_cast<float>(rules.height + 2 - 2 * canonical_ball.y) /
static_cast<float>(rules.height + 2);
const int mover_distance = mover == Player::One ? player_one_distance
: player_two_distance;
const int opponent_distance = mover == Player::One ? player_two_distance
: player_one_distance;
snapshot.features[13] =
static_cast<float>(std::min(mover_distance, 16)) / 16.0F;
snapshot.features[14] =
static_cast<float>(std::min(opponent_distance, 16)) / 16.0F;
snapshot.features[15] =
static_cast<float>(std::clamp(sign * distance_advantage, -16, 16)) /
16.0F;
snapshot.features[16] = static_cast<float>(count) / 8.0F;
snapshot.features[17] = static_cast<float>(continuations) / 8.0F;
snapshot.features[18] = static_cast<float>(forward_moves) / 8.0F;
snapshot.features[19] =
static_cast<float>(center_alignment) /
static_cast<float>(rules.width / 2);
snapshot.features[20] = direct_goal ? 1.0F : 0.0F;
snapshot.features[21] = is_boundary_point(rules, ball) ? 1.0F : 0.0F;
snapshot.features[22] = count == 1 ? 1.0F : 0.0F;
snapshot.features[23] =
static_cast<float>(std::min<int>(replay.used_edges, 64)) / 64.0F;
return snapshot;
}

float teacher_residual_prediction(
const std::array<float, kTeacherResidualInputs> &features) const {
static_assert(teacher_residual_model::kInputCount ==
kTeacherResidualInputs);
float prediction = teacher_residual_model::kBias;
for (std::size_t input = 0; input < features.size(); ++input) {
prediction += features[input] * teacher_residual_model::kWeights[input];
}
return std::clamp(prediction, -1.0F, 1.0F);
}

int evaluate() {
++stats_.leaf_evaluations;
const EvaluationSnapshot snapshot = make_evaluation_snapshot();
int score = snapshot.anchor_score;
if (config_.teacher_residual_weight_percent != 0 &&
teacher_residual_root_enabled_) {
++stats_.teacher_residual_evaluations;
const int mover_correction = static_cast<int>(std::lround(
teacher_residual_prediction(snapshot.features) *
kTeacherResidualTargetScale *
config_.teacher_residual_weight_percent / 100.0F));
const int confidence_cap =
std::max(0, kTeacherResidualHardCap -
std::abs(snapshot.anchor_score) / 10);
const int correction =
snapshot.features[20] != 0.0F ||
snapshot.features[23] * 64.0F <
kTeacherResidualMinimumUsedEdges
? 0
: snapshot.mover_sign *
std::clamp(mover_correction, -confidence_cap,
confidence_cap);
score += correction;
}
return std::clamp(score, -kMaximumEvaluation, kMaximumEvaluation);
}

int cached_evaluate() {
const detail::PositionKey key = boundary_key();
++stats_.evaluation_cache_probes;
if (const std::optional<int> cached = evaluations_.find(key)) {
++stats_.evaluation_cache_hits;
return *cached;
}
const int score = evaluate();
evaluations_.store(key, score);
return score;
}

OrderedMoveList ordered_moves(
std::optional<Move> preferred = std::nullopt) {
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = position_.legal_slots(slots);
OrderedMoveList ordered;
ordered.count = count;
const Player mover = position_.to_move();
const Point ball = position_.ball();
const auto source = position_.ball_vertex();
const RulesConfig &rules = topology_->config();

for (std::uint8_t index = 0; index < count; ++index) {
const std::uint8_t slot = slots[index];
const Move move = position_.move_for_slot(slot);
const bool extra_turn = position_.grants_extra_turn(slot);
int score = extra_turn ? 10'000 : 0;
const int progress = mover == Player::One ? ball.y - move.to.y
: move.to.y - ball.y;
score += progress * 100;
score -= std::abs(move.to.x - topology_->config().width / 2) * 4;
if (preferred.has_value() && move == *preferred) {
score += 2'000'000;
}

const detail::SearchTopology::Arc &played_arc =
topology_->adjacency(source, mover).arcs[slot];
if (is_goal_point(rules, move.to)) {
const Player winning_player =
is_attacking_goal(rules, move.to, Player::One) ? Player::One
: Player::Two;
score += winning_player == mover ? 1'000'000 : -1'000'000;
} else {
const Player child_player = extra_turn ? mover : opponent(mover);
const detail::SearchTopology::Adjacency &child_adjacency =
topology_->adjacency(played_arc.destination, child_player);
int mobility = 0;
for (std::uint8_t child = 0; child < child_adjacency.count; ++child) {
const auto edge = child_adjacency.arcs[child].edge;
mobility += edge != played_arc.edge && !position_.edge_used(edge)
? 1
: 0;
}
if (mobility == 0) {
const Player blocked_player =
rules.blocked_rule == BlockedRule::MoverLoses ? mover
: child_player;
score += opponent(blocked_player) == mover ? 1'000'000
: -1'000'000;
} else {
score += child_player == mover ? mobility * 12 : -mobility * 12;
}
}
ordered.values[index] = OrderedMove{slot, move, score};
}

const auto better = [mover](const OrderedMove &left,
const OrderedMove &right) {
if (left.score != right.score) {
return left.score > right.score;
}
if (left.move.to.y != right.move.to.y) {
return mover == Player::One ? left.move.to.y < right.move.to.y
: left.move.to.y > right.move.to.y;
}
return mover == Player::One ? left.move.to.x < right.move.to.x
: left.move.to.x > right.move.to.x;
};
for (std::uint8_t index = 1; index < count; ++index) {
const OrderedMove value = ordered.values[index];
std::uint8_t insertion = index;
while (insertion > 0 &&
better(value, ordered.values[insertion - 1U])) {
ordered.values[insertion] = ordered.values[insertion - 1U];
--insertion;
}
ordered.values[insertion] = value;
}
return ordered;
}

std::vector<Move> fallback_action() {
const std::size_t root_depth = position_.undo_depth();
const Player mover = position_.to_move();
std::vector<Move> action;
try {
while (!position_.is_terminal() && position_.to_move() == mover) {
const OrderedMoveList moves = ordered_moves();
if (moves.count == 0) {
throw std::logic_error("fallback reached a state without moves");
}
action.push_back(moves.values[0].move);
position_.make_move(moves.values[0].slot);
}
} catch (...) {
position_.unmake_to(root_depth);
throw;
}
position_.unmake_to(root_depth);
return action;
}

void record_completed_action(int score, Player root_mover) {
++stats_.completed_actions;
stats_.max_action_edges = std::max<std::uint32_t>(
stats_.max_action_edges,
static_cast<std::uint32_t>(current_action_.size()));
const bool better =
!captured_any_ ||
(root_mover == Player::One ? score > captured_score_
: score < captured_score_);
if (better) {
captured_any_ = true;
captured_score_ = score;
captured_action_ = current_action_;
}
}

bool reuse_cached_root_action(Player mover, std::uint32_t remaining_depth,
int score) {
const std::size_t base_position_depth = position_.undo_depth();
const std::size_t base_action_size = current_action_.size();
try {
while (!position_.is_terminal() && position_.to_move() == mover) {
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = position_.legal_slots(slots);
if (count == 0) {
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
return false;
}

std::uint8_t slot = slots[0];
if (count > 1) {
const TranspositionEntry *entry = table_.find(boundary_key());
if (entry == nullptr || entry->depth < remaining_depth ||
entry->bound != ScoreBound::Exact ||
!position_.slot_for_move(entry->best_move, slot)) {
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
return false;
}
}
current_action_.push_back(position_.move_for_slot(slot));
position_.make_move(slot);
}
record_completed_action(score, mover);
++stats_.root_transposition_reuses;
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
return true;
} catch (...) {
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
throw;
}
}

int explore_continuation(Player mover, std::uint32_t remaining_depth,
std::uint32_t turn_ply, int alpha, int beta,
bool capture_root) {
const std::size_t base_position_depth = position_.undo_depth();
const std::size_t base_action_size = current_action_.size();
try {
visit_node();
const detail::PositionKey key = boundary_key();
const int original_alpha = alpha;
const int original_beta = beta;
std::optional<Move> preferred;
if (const TranspositionEntry *entry = probe(key)) {
++stats_.continuation_transposition_hits;
preferred = entry->best_move;
if (capture_root && config_.root_transposition_pruning &&
entry->depth >= remaining_depth &&
entry->bound == ScoreBound::Exact &&
reuse_cached_root_action(mover, remaining_depth, entry->score)) {
++stats_.transposition_cutoffs;
return entry->score;
}
if (entry->depth >= remaining_depth) {
if (entry->bound == ScoreBound::Exact) {
const bool cannot_improve =
mover == Player::One ? entry->score <= alpha
: entry->score >= beta;
if (!capture_root ||
(config_.root_transposition_pruning && cannot_improve)) {
++stats_.transposition_cutoffs;
return entry->score;
}
} else if (!capture_root || config_.root_transposition_pruning) {
if (entry->bound == ScoreBound::Lower) {
alpha = std::max(alpha, entry->score);
} else {
beta = std::min(beta, entry->score);
}
if (alpha >= beta) {
++stats_.transposition_cutoffs;
return entry->score;
}
}
}
}

OrderedMoveList moves = ordered_moves(preferred);
std::optional<Move> first_forced_move;
while (moves.count == 1) {
const OrderedMove forced = moves.values[0];
if (!first_forced_move.has_value()) {
first_forced_move = forced.move;
}
position_.make_move(forced.slot);
current_action_.push_back(forced.move);
++stats_.forced_edges;
if (position_.is_terminal() || position_.to_move() != mover) {
const int score =
search(remaining_depth - 1U, turn_ply + 1U, alpha, beta);
if (capture_root) {
record_completed_action(score, mover);
}
ScoreBound bound = ScoreBound::Exact;
if (score <= original_alpha) {
bound = ScoreBound::Upper;
} else if (score >= original_beta) {
bound = ScoreBound::Lower;
}
store(key, remaining_depth, score, *first_forced_move, bound);
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
return score;
}
visit_node();
moves = ordered_moves();
}

const bool maximizing = mover == Player::One;
int best_score = maximizing ? -kInfinity : kInfinity;
std::optional<Move> best_move = first_forced_move;
bool found = false;
for (std::uint8_t index = 0; index < moves.count; ++index) {
const OrderedMove &candidate = moves.values[index];
int score = 0;
{
ScopedMove played(position_, candidate.slot);
current_action_.push_back(candidate.move);
if (position_.is_terminal() || position_.to_move() != mover) {
score = search(remaining_depth - 1U, turn_ply + 1U, alpha, beta);
if (capture_root) {
record_completed_action(score, mover);
}
} else {
score = explore_continuation(mover, remaining_depth, turn_ply,
alpha, beta, capture_root);
}
current_action_.pop_back();
}
if (!found ||
(maximizing ? score > best_score : score < best_score)) {
best_score = score;
if (!first_forced_move.has_value()) {
best_move = candidate.move;
}
}
found = true;
if (maximizing) {
alpha = std::max(alpha, best_score);
} else {
beta = std::min(beta, best_score);
}
if (reached_immediate_win_bound(best_score, mover, turn_ply)) {
++stats_.terminal_bound_cutoffs;
break;
}
if (alpha >= beta) {
++stats_.cutoffs;
break;
}
}
if (!found || !best_move.has_value()) {
throw std::logic_error("continuation node has no legal edge");
}
ScoreBound bound = ScoreBound::Exact;
if (best_score <= original_alpha) {
bound = ScoreBound::Upper;
} else if (best_score >= original_beta) {
bound = ScoreBound::Lower;
}
store(key, remaining_depth, best_score, *best_move, bound);
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
return best_score;
} catch (...) {
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
throw;
}
}

int seed_root_endpoint(const OrderedMove &first, std::uint32_t turn_ply,
int alpha, int beta, Player mover) {
const std::size_t base_position_depth = position_.undo_depth();
const std::size_t base_action_size = current_action_.size();
try {
position_.make_move(first.slot);
current_action_.push_back(first.move);
while (!position_.is_terminal() && position_.to_move() == mover) {
visit_node();
const OrderedMoveList moves = ordered_moves();
if (moves.count == 0) {
throw std::logic_error("root seed reached a state without moves");
}
position_.make_move(moves.values[0].slot);
current_action_.push_back(moves.values[0].move);
}
const int score = search(0, turn_ply + 1U, alpha, beta);
record_completed_action(score, mover);
++stats_.root_seed_actions;
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
return score;
} catch (...) {
position_.unmake_to(base_position_depth);
current_action_.resize(base_action_size);
throw;
}
}

ActionSearchResult search_actions(
std::uint32_t remaining_depth, std::uint32_t turn_ply, int alpha,
int beta, std::optional<Move> preferred, bool capture_root) {
const Player mover = position_.to_move();
const bool maximizing = mover == Player::One;
int best_score = maximizing ? -kInfinity : kInfinity;
std::optional<Move> best_move;
OrderedMoveList moves = ordered_moves(preferred);

if (capture_root && remaining_depth == 1U &&
config_.root_seed_endpoints) {
for (std::uint8_t index = 0; index < moves.count; ++index) {
const int score = seed_root_endpoint(moves.values[index], turn_ply,
alpha, beta, mover);
moves.values[index].score = mover == Player::One ? score : -score;
if (reached_immediate_win_bound(score, mover, turn_ply)) {
++stats_.terminal_bound_cutoffs;
return ActionSearchResult{score, captured_action_.front()};
}
}
for (std::uint8_t index = 1; index < moves.count; ++index) {
const OrderedMove value = moves.values[index];
std::uint8_t insertion = index;
while (insertion > 0 &&
value.score > moves.values[insertion - 1U].score) {
moves.values[insertion] = moves.values[insertion - 1U];
--insertion;
}
moves.values[insertion] = value;
}
}

for (std::uint8_t index = 0; index < moves.count; ++index) {
const OrderedMove &candidate = moves.values[index];
int score = 0;
{
ScopedMove played(position_, candidate.slot);
current_action_.push_back(candidate.move);
if (position_.is_terminal() || position_.to_move() != mover) {
score = search(remaining_depth - 1U, turn_ply + 1U, alpha, beta);
if (capture_root) {
record_completed_action(score, mover);
}
} else {
score = explore_continuation(mover, remaining_depth, turn_ply,
alpha, beta, capture_root);
}
current_action_.pop_back();
}

if (!best_move.has_value() ||
(maximizing ? score > best_score : score < best_score)) {
best_score = score;
best_move = candidate.move;
}
if (maximizing) {
alpha = std::max(alpha, best_score);
} else {
beta = std::min(beta, best_score);
}
if (reached_immediate_win_bound(best_score, mover, turn_ply)) {
++stats_.terminal_bound_cutoffs;
break;
}
if (!capture_root && alpha >= beta) {
++stats_.cutoffs;
break;
}
}
if (!best_move.has_value()) {
throw std::logic_error("turn boundary has no legal action");
}
return ActionSearchResult{best_score, *best_move};
}

const TranspositionEntry *probe(detail::PositionKey key) {
++stats_.transposition_probes;
const TranspositionEntry *entry = table_.find(key);
if (entry != nullptr) {
++stats_.transposition_hits;
}
return entry;
}

void store(detail::PositionKey key, std::uint32_t depth, int score,
Move best_move, ScoreBound bound) {
if (table_.store(key, depth, score, best_move, bound)) {
++stats_.transposition_stores;
}
}

int search(std::uint32_t remaining_depth, std::uint32_t turn_ply,
int alpha, int beta) {
visit_node();
if (position_.is_terminal()) {
return terminal_score(turn_ply);
}
if (remaining_depth == 0) {
if (!exact_proof_enabled(kExactProofLeafBoundary)) {
return cached_evaluate();
}
const detail::PositionKey key = boundary_key();
++stats_.evaluation_cache_probes;
if (const std::optional<int> cached = evaluations_.find(key)) {
++stats_.evaluation_cache_hits;
return *cached;
}
++stats_.leaf_rebound_probes;
const ReboundOutcome rebound = analyze_rebound_component(nullptr);
if (rebound == ReboundOutcome::Win) {
++stats_.leaf_rebound_win_hits;
return immediate_win_score(position_.to_move(), turn_ply);
}
if (rebound == ReboundOutcome::Loss) {
++stats_.leaf_rebound_loss_hits;
return immediate_win_score(opponent(position_.to_move()), turn_ply);
}
const int score = evaluate();
evaluations_.store(key, score);
return score;
}

const detail::PositionKey key = boundary_key();
const int original_alpha = alpha;
const int original_beta = beta;
std::optional<Move> preferred;
if (const TranspositionEntry *entry = probe(key)) {
preferred = entry->best_move;
if (entry->depth >= remaining_depth) {
if (entry->bound == ScoreBound::Exact) {
++stats_.transposition_cutoffs;
return entry->score;
}
if (entry->bound == ScoreBound::Lower) {
alpha = std::max(alpha, entry->score);
} else {
beta = std::min(beta, entry->score);
}
if (alpha >= beta) {
++stats_.transposition_cutoffs;
return entry->score;
}
}
}

if ((turn_ply == 1U && exact_proof_enabled(kExactProofPlyOne)) ||
(turn_ply == 2U && exact_proof_enabled(kExactProofPlyTwo))) {
std::uint64_t &probes = turn_ply == 1U
? stats_.exchange_ply1_probes
: stats_.exchange_ply2_probes;
std::uint64_t &win_hits = turn_ply == 1U
? stats_.exchange_ply1_win_hits
: stats_.exchange_ply2_win_hits;
std::uint64_t &loss_hits = turn_ply == 1U
? stats_.exchange_ply1_loss_hits
: stats_.exchange_ply2_loss_hits;
std::uint64_t &cutoffs = turn_ply == 1U
? stats_.exchange_ply1_cutoffs
: stats_.exchange_ply2_cutoffs;
++probes;
const ReboundOutcome rebound = analyze_rebound_component(nullptr);
if (rebound == ReboundOutcome::Win) {
++win_hits;
++cutoffs;
return immediate_win_score(position_.to_move(), turn_ply);
}
if (rebound == ReboundOutcome::Loss) {
++loss_hits;
++cutoffs;
return immediate_win_score(opponent(position_.to_move()), turn_ply);
}
}

const ActionSearchResult result = search_actions(
remaining_depth, turn_ply, alpha, beta, preferred, false);
ScoreBound bound = ScoreBound::Exact;
if (result.score <= original_alpha) {
bound = ScoreBound::Upper;
} else if (result.score >= original_beta) {
bound = ScoreBound::Lower;
}
store(key, remaining_depth, result.score, result.first_move, bound);
return result.score;
}

RootResult search_root(std::uint32_t depth) {
visit_node();
const detail::PositionKey key = boundary_key();
std::optional<Move> preferred;
if (const TranspositionEntry *entry = probe(key)) {
preferred = entry->best_move;
}
current_action_.clear();
captured_action_.clear();
captured_any_ = false;
const ActionSearchResult result =
search_actions(depth, 0, -kInfinity, kInfinity, preferred, true);
if (!captured_any_ || captured_action_.empty()) {
throw std::logic_error("root search did not complete a legal action");
}
store(key, depth, result.score, captured_action_.front(),
ScoreBound::Exact);
return RootResult{result.score, captured_action_};
}
};

Player player_for_id(int player_id) {
if (player_id == 0) {
return Player::One;
}
if (player_id == 1) {
return Player::Two;
}
throw std::invalid_argument("player id must be zero or one");
}

Move decode_direction(Point from, char direction) {
if (direction < '0' || direction > '7') {
throw std::invalid_argument("direction must be between zero and seven");
}
const Point delta =
kDirectionDeltas[static_cast<std::size_t>(direction - '0')];
return Move{{from.x + delta.x, from.y + delta.y}};
}

char encode_direction(Point from, Point to) {
const Point delta{to.x - from.x, to.y - from.y};
for (std::size_t index = 0; index < kDirectionDeltas.size(); ++index) {
if (kDirectionDeltas[index] == delta) {
return static_cast<char>('0' + index);
}
}
throw std::invalid_argument("move is not an adjacent direction");
}

void apply_encoded_turn(GameState &state, std::string_view encoded) {
if (encoded.empty()) {
throw std::invalid_argument("a turn cannot be empty");
}
GameState next = state;
const Player mover = next.to_move;
for (const char direction : encoded) {
if (is_terminal(next) || next.to_move != mover) {
throw std::invalid_argument("encoded action continues after turn end");
}
next = apply_move(next, decode_direction(next.ball, direction));
}
if (!is_terminal(next) && next.to_move == mover) {
throw std::invalid_argument("encoded action omits a required rebound");
}
state = std::move(next);
}

bool lookup_replay_correction(int player_id, std::string_view transcript,
std::string &encoded) {
const std::size_t completed_turns =
transcript.empty()
? 0U
: 1U + static_cast<std::size_t>(
std::count(transcript.begin(), transcript.end(), '/'));
for (const replay_book::Replay &replay : replay_book::kReplays) {
if (replay.player_id != player_id ||
completed_turns < replay.first_turn ||
completed_turns % 2U != static_cast<std::size_t>(player_id)) {
continue;
}

const std::string_view recorded = replay.transcript;
std::size_t action_begin = 0;
if (transcript.empty()) {
if (replay.first_turn != 0) {
continue;
}
} else {
if (recorded.size() <= transcript.size() ||
!recorded.starts_with(transcript) ||
recorded[transcript.size()] != '/') {
continue;
}
action_begin = transcript.size() + 1U;
}
const std::size_t action_end = recorded.find('/', action_begin);
encoded.assign(
recorded.substr(action_begin, action_end - action_begin));
if (!encoded.empty()) {
return true;
}
}
return false;
}

bool try_replay_correction(GameState &state, int player_id,
std::string_view transcript,
std::string &encoded) {
std::string candidate;
if (!lookup_replay_correction(player_id, transcript, candidate)) {
return false;
}

GameState next = state;
try {
apply_encoded_turn(next, candidate);
} catch (const std::exception &) {
return false;
}

state = std::move(next);
encoded = std::move(candidate);
return true;
}

std::string choose_complete_turn(GameState &state,
std::uint32_t search_time_ms) {
SearchConfig config;
config.exact_proof_mask = kAllExactProofs;
config.replay_value_blend_percent = 15;
config.teacher_residual_weight_percent = 100;
config.absolute_deadline =
SearchClock::now() + std::chrono::milliseconds(search_time_ms);
CompleteTurnSearch search(state, config);
const std::vector<Move> action = search.run();
if (action.empty()) {
throw std::logic_error("search returned an empty action");
}

const Player mover = state.to_move;
std::string encoded;
encoded.reserve(action.size());
for (const Move move : action) {
if (is_terminal(state) || state.to_move != mover) {
throw std::logic_error("search action continues after turn end");
}
encoded.push_back(encode_direction(state.ball, move.to));
state = apply_move(state, move);
}
if (!is_terminal(state) && state.to_move == mover) {
throw std::logic_error("search action ends before rebound completion");
}
return encoded;
}

}  // namespace papersoccer::turn_action_v2

#ifndef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
int main() {
using namespace papersoccer;
using namespace papersoccer::turn_action_v2;

std::ios::sync_with_stdio(false);
std::cin.tie(nullptr);

int player_id = -1;
if (!(std::cin >> player_id)) {
return 0;
}
Player me;
try {
me = player_for_id(player_id);
} catch (const std::exception &) {
return 1;
}
std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

RulesConfig rules;
rules.goal_rule = GoalRule::OwnGoalsAllowed;
rules.blocked_rule = BlockedRule::MoverLoses;
GameState state = make_initial_state(rules);
bool first_execution = true;
std::string transcript;

while (true) {
int opponent_move_length = 0;
if (!(std::cin >> opponent_move_length)) {
break;
}
std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
std::string opponent_move;
if (!std::getline(std::cin, opponent_move)) {
break;
}
if (opponent_move != "-" &&
opponent_move_length != static_cast<int>(opponent_move.size())) {
return 1;
}

try {
if (opponent_move == "-") {
if (player_id != 0 || !first_execution) {
return 1;
}
} else {
apply_encoded_turn(state, opponent_move);
if (!transcript.empty()) {
transcript.push_back('/');
}
transcript += opponent_move;
}
if (is_terminal(state)) {
return 0;
}
if (state.to_move != me) {
return 1;
}

const std::uint32_t time_ms =
first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs;
std::string action;
if (!try_replay_correction(state, player_id, transcript, action)) {
action = choose_complete_turn(state, time_ms);
}
std::cout << action << std::endl;
if (!transcript.empty()) {
transcript.push_back('/');
}
transcript += action;
first_execution = false;
} catch (const std::exception &) {
return 1;
}
}
return 0;
}
#endif
