
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
inline constexpr std::string_view kEncodedWeights = "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";

}  // namespace papersoccer::turn_action_v2::replay_value_model

namespace papersoccer::turn_action_v2::teacher_residual_model {

inline constexpr std::size_t kInputCount = 24;
inline constexpr std::array<float, kInputCount> kWeights{{-0.000212487676F, -0.0314205534F, 0.00000000F, 0.0220964056F, 0.00318384444F, -0.0459056867F, 0.00000000F, 0.00000000F, -0.000812118322F, -0.0889680974F, -0.0757096574F, 0.00821899013F, -0.163250396F, -0.206006910F, 0.0891054922F, 0.00699152156F, -0.000185926722F, 0.0146061040F, 0.127743796F, 0.0246996844F, 0.00000000F, 0.00000000F, 0.00000000F, 0.00474756504F}};
inline constexpr float kBias = -0.0212487676F;

}  // namespace papersoccer::turn_action_v2::teacher_residual_model

namespace papersoccer::turn_action_v2 {

constexpr std::uint32_t kFirstSearchTimeMs = 800;
constexpr std::uint32_t kLaterSearchTimeMs = 165;
constexpr std::uint64_t kMaximumWork = 3'000'000;
constexpr std::size_t kMaximumTreeNodes = 120'000;
constexpr std::size_t kMaximumActions = 250;
constexpr std::uint64_t kMaximumPartialPaths = 50'000;
constexpr std::uint64_t kTacticalPartialPaths = 32;
constexpr std::size_t kEvaluationEntries = 131'072;
constexpr double kExplorationConstant = 1.5;
constexpr double kFirstPlayUrgency = 0.5;

constexpr int kMateScore = 1'000'000;
constexpr int kMaximumMateDistance = 32;
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
{0, -1}, {1, -1}, {1, 0}, {1, 1},
{0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
}};

using SearchClock = std::chrono::steady_clock;

struct SearchConfig {
std::uint64_t max_work{kMaximumWork};
std::size_t max_tree_nodes{kMaximumTreeNodes};
std::size_t max_actions{kMaximumActions};
std::uint64_t max_partial_paths{kMaximumPartialPaths};
std::size_t evaluation_entries{kEvaluationEntries};
std::uint32_t max_time_ms{};
std::optional<SearchClock::time_point> absolute_deadline{};
int replay_value_blend_percent{};
int teacher_residual_weight_percent{};
double exploration_constant{kExplorationConstant};
double first_play_urgency{kFirstPlayUrgency};
};

struct SearchStats {
std::uint64_t work{};
std::uint64_t tree_nodes{};
std::uint64_t expansions{};
std::uint64_t child_evaluations{};
std::uint64_t generator_partial_paths{};
std::uint64_t completed_actions{};
std::uint64_t generator_duplicates{};
std::uint64_t tactical_actions{};
std::uint64_t generator_truncations{};
std::uint32_t max_deque_size{};
std::uint32_t max_action_edges{};
int root_score{};
bool budget_exhausted{};
bool deadline_reached{};
bool node_cap_reached{};
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
entries_[bucket + ((combined_key(key) >> 32U) & 1U)] =
EvaluationEntry{key, score, true};
}

private:
std::vector<EvaluationEntry> entries_;

static std::uint64_t combined_key(detail::PositionKey key) noexcept {
return key.first ^ ((key.second << 29U) | (key.second >> 35U));
}

std::size_t bucket_index(detail::PositionKey key) const noexcept {
return 2U * static_cast<std::size_t>(
combined_key(key) % (entries_.size() / 2U));
}
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

Player player_for_id(int player_id) {
if (player_id == 0) {
return Player::One;
}
if (player_id == 1) {
return Player::Two;
}
throw std::invalid_argument("");
}

Move decode_direction(Point from, char direction) {
if (direction < '0' || direction > '7') {
throw std::invalid_argument("");
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
throw std::invalid_argument("");
}

detail::PositionKey boundary_key(
const detail::SearchPosition &position) noexcept {
detail::PositionKey key = position.position_key();
detail::xor_position_key(
key, detail::position_key_component(6, position.ball_vertex()));
return key;
}

class SplitMix64 {
public:
explicit SplitMix64(std::uint64_t state) : state_(state) {}

std::uint64_t next() noexcept {
state_ += 0x9e3779b97f4a7c15ULL;
return detail::mix_position_key(state_);
}

std::size_t index(std::size_t bound) noexcept {
if (bound <= 1) {
return 0;
}
const std::uint64_t threshold =
static_cast<std::uint64_t>(-bound) % bound;
for (;;) {
const std::uint64_t value = next();
if (value >= threshold) {
return static_cast<std::size_t>(value % bound);
}
}
}

private:
std::uint64_t state_{};
};

enum class TacticalClass {
ImmediateWin,
ForcedCutoff,
SafeHandoff,
OpponentImmediateWin,
TerminalLoss,
};

struct GeneratorConfig {
std::size_t max_actions{kMaximumActions};
std::uint64_t max_partial_paths{kMaximumPartialPaths};
std::uint64_t max_work{kMaximumWork};
std::optional<SearchClock::time_point> absolute_deadline{};
};

struct GeneratorStats {
std::uint64_t explored_partial_paths{};
std::uint64_t completed_actions{};
std::uint64_t duplicates{};
std::uint64_t tactical_actions{};
std::uint64_t truncations{};
std::uint64_t fifo_extractions{};
std::uint64_t lifo_extractions{};
std::uint64_t immediate_wins{};
std::uint64_t terminal_losses{};
std::uint64_t safe_handoffs{};
std::uint64_t opponent_immediate_wins{};
std::uint64_t forced_cutoffs{};
std::uint32_t max_deque_size{};
std::uint32_t max_action_edges{};
bool deadline_reached{};
};

struct CompleteTurnAction {
std::vector<Move> moves;
detail::SearchPosition result;
TacticalClass tactical{TacticalClass::SafeHandoff};
std::string encoded;

CompleteTurnAction(std::vector<Move> action,
detail::SearchPosition position,
TacticalClass classification, std::string key)
: moves(std::move(action)), result(std::move(position)),
tactical(classification), encoded(std::move(key)) {}
};

struct GenerationResult {
std::vector<CompleteTurnAction> actions;
GeneratorStats stats{};
bool exhaustive{};
};

enum class ReboundOutcome { Unknown, Win, Loss };

class FullTurnGenerator {
public:
FullTurnGenerator(std::shared_ptr<const detail::SearchTopology> topology,
const detail::SearchPosition &root,
GeneratorConfig config = {})
: topology_(std::move(topology)), root_(root), config_(config),
parents_(topology_->vertex_count(), kUnseen),
queue_(topology_->vertex_count()) {
if (config_.max_actions == 0 ||
config_.max_actions > kMaximumActions ||
config_.max_partial_paths == 0 ||
config_.max_partial_paths > kMaximumPartialPaths ||
config_.max_work == 0 || config_.max_work > kMaximumWork) {
throw std::invalid_argument("");
}
edge_vertices_.resize(topology_->edge_count());
for (std::uint32_t vertex = 0; vertex < topology_->vertex_count();
++vertex) {
for (const Player player : {Player::One, Player::Two}) {
const auto &adjacency = topology_->adjacency(vertex, player);
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
edge_vertices_[arc.edge] = {vertex, arc.destination};
}
}
}
}

GenerationResult run() {
if (root_.is_terminal()) {
throw std::invalid_argument("");
}
retained_keys_.clear();
deadline_reached_ = false;

GenerationResult output;
output.actions.reserve(config_.max_actions);
bool truncated = false;
if (!deadline_expired()) {
retain_exact_goal_path(output, root_.to_move());
}
if (!deadline_expired()) {
retain_bounded_handoff_paths(output);
}
if (!deadline_expired()) {
retain_exact_goal_path(output, opponent(root_.to_move()));
}
if (deadline_reached_) {
output.stats.deadline_reached = true;
truncated = true;
}

std::deque<Partial> pending;
if (!deadline_reached_) {
pending.emplace_back(root_, std::vector<Move>{});
output.stats.max_deque_size =
std::max<std::uint32_t>(output.stats.max_deque_size, 1);
}

while (!pending.empty()) {
if (output.actions.size() >= config_.max_actions ||
output.stats.explored_partial_paths >=
config_.max_partial_paths ||
output.stats.explored_partial_paths >= config_.max_work) {
truncated = true;
break;
}
if (deadline_expired()) {
truncated = true;
output.stats.deadline_reached = true;
break;
}

++output.stats.explored_partial_paths;
Partial partial = [&] {
if (output.stats.explored_partial_paths % 10U == 0U) {
++output.stats.fifo_extractions;
Partial value = std::move(pending.front());
pending.pop_front();
return value;
}
++output.stats.lifo_extractions;
Partial value = std::move(pending.back());
pending.pop_back();
return value;
}();

std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = partial.position.legal_slots(slots);
shuffle(slots, count, partial.position);
for (std::uint8_t index = 0; index < count; ++index) {
Partial child = partial;
const Move move = child.position.move_for_slot(slots[index]);
child.moves.push_back(move);
child.position.make_move(slots[index]);
if (child.position.is_terminal() ||
child.position.to_move() != root_.to_move()) {
retain(output, std::move(child.moves),
std::move(child.position));
if (output.actions.size() >= config_.max_actions) {
truncated = truncated || !pending.empty() || index + 1U < count;
break;
}
} else {
if (pending.size() < config_.max_partial_paths) {
pending.push_back(std::move(child));
output.stats.max_deque_size = std::max<std::uint32_t>(
output.stats.max_deque_size,
static_cast<std::uint32_t>(pending.size()));
} else {
truncated = true;
}
}
}
if (output.actions.size() >= config_.max_actions) {
break;
}
}

if (deadline_reached_) {
output.stats.deadline_reached = true;
truncated = true;
}

if (output.actions.empty() && !deadline_reached_) {
Partial fallback{root_, fallback_action(root_)};
fallback.position = root_;
for (const Move move : fallback.moves) {
std::uint8_t slot = 0;
if (!fallback.position.slot_for_move(move, slot)) {
throw std::logic_error("");
}
fallback.position.make_move(slot);
}
retain(output, std::move(fallback.moves),
std::move(fallback.position));
truncated = true;
}

std::stable_sort(output.actions.begin(), output.actions.end(),
[](const CompleteTurnAction &left,
const CompleteTurnAction &right) {
if (left.tactical != right.tactical) {
return static_cast<int>(left.tactical) <
static_cast<int>(right.tactical);
}
return left.encoded < right.encoded;
});
output.exhaustive = pending.empty() && !truncated;
if (truncated) {
++output.stats.truncations;
}
return output;
}

private:
static constexpr int kUnseen = -2;

struct Partial {
detail::SearchPosition position;
std::vector<Move> moves;

Partial(detail::SearchPosition state, std::vector<Move> action)
: position(std::move(state)), moves(std::move(action)) {}
};

std::shared_ptr<const detail::SearchTopology> topology_;
detail::SearchPosition root_;
GeneratorConfig config_;
std::vector<int> parents_;
std::vector<detail::SearchTopology::VertexIndex> queue_;
std::vector<detail::PositionKey> retained_keys_;
std::vector<std::pair<detail::SearchTopology::VertexIndex,
detail::SearchTopology::VertexIndex>> edge_vertices_;
bool deadline_reached_{};

bool deadline_expired() {
if (!deadline_reached_ && config_.absolute_deadline.has_value() &&
SearchClock::now() >= *config_.absolute_deadline) {
deadline_reached_ = true;
}
return deadline_reached_;
}

Point transformed(Point point, unsigned symmetry) const noexcept {
if ((symmetry & 2U) != 0U) {
point = {topology_->config().width - point.x,
topology_->config().height + 2 - point.y};
}
if ((symmetry & 1U) != 0U) {
point.x = topology_->config().width - point.x;
}
return point;
}

static std::uint64_t packed(Point point) noexcept {
return static_cast<std::uint64_t>(point.y * 16 + point.x);
}

std::uint64_t symmetry_key(const detail::SearchPosition &position,
unsigned symmetry) const noexcept {
std::uint64_t key = 0;
const auto add = [&](std::uint64_t category, std::uint64_t value) {
key ^= detail::mix_position_key((category << 56U) ^ value ^
0xd1b54a32d192ed03ULL);
};
for (std::uint32_t vertex = 0; vertex < topology_->vertex_count();
++vertex) {
if (position.vertex_visited(vertex)) {
add(2, packed(transformed(topology_->point(vertex), symmetry)));
}
}
for (std::uint32_t edge = 0; edge < topology_->edge_count(); ++edge) {
if (position.edge_used(edge)) {
auto [first, second] = edge_vertices_[edge];
std::uint64_t a = packed(transformed(topology_->point(first), symmetry));
std::uint64_t b = packed(transformed(topology_->point(second), symmetry));
if (a > b) std::swap(a, b);
add(1, (a << 8U) | b);
}
}
Player mover = position.to_move();
Status status = position.status();
if ((symmetry & 2U) != 0U) {
mover = opponent(mover);
if (status == Status::WonByOne) status = Status::WonByTwo;
else if (status == Status::WonByTwo) status = Status::WonByOne;
}
add(3, packed(transformed(position.ball(), symmetry)));
add(4, static_cast<std::uint64_t>(mover));
add(5, static_cast<std::uint64_t>(status));
return key;
}

std::pair<std::uint64_t, unsigned> canonical_key(
const detail::SearchPosition &position) const noexcept {
std::pair<std::uint64_t, unsigned> best{symmetry_key(position, 0), 0};
for (unsigned symmetry = 1; symmetry < 4; ++symmetry) {
const std::uint64_t key = symmetry_key(position, symmetry);
if (key < best.first) {
best = {key, symmetry};
}
}
return best;
}

static unsigned canonical_direction(Point from, Point to,
unsigned symmetry) {
unsigned value =
static_cast<unsigned>(encode_direction(from, to) - '0');
if ((symmetry & 2U) != 0U) value = (value + 4U) & 7U;
if ((symmetry & 1U) != 0U) value = (8U - value) & 7U;
return value;
}

void shuffle(std::array<std::uint8_t, detail::kMaximumMoves> &slots,
std::uint8_t count,
const detail::SearchPosition &position) const {
const auto [key, symmetry] = canonical_key(position);
std::sort(slots.begin(), slots.begin() + count,
[&](std::uint8_t left, std::uint8_t right) {
return canonical_direction(
position.ball(), position.move_for_slot(left).to,
symmetry) <
canonical_direction(
position.ball(), position.move_for_slot(right).to,
symmetry);
});
SplitMix64 random(key);
for (std::size_t remaining = count; remaining > 1; --remaining) {
std::swap(slots[remaining - 1U],
slots[random.index(remaining)]);
}
if (symmetry_key(position, 0) == symmetry_key(position, 1)) {
for (std::uint8_t index = 0; index < count; ++index) {
const unsigned direction = canonical_direction(
position.ball(), position.move_for_slot(slots[index]).to, 0);
if (direction == 0 || direction == 4) {
std::swap(slots[0], slots[index]);
break;
}
}
}
}

std::string encode_action(const std::vector<Move> &moves) const {
detail::SearchPosition position = root_;
std::string encoded;
encoded.reserve(moves.size());
for (const Move move : moves) {
encoded.push_back(encode_direction(position.ball(), move.to));
std::uint8_t slot = 0;
if (!position.slot_for_move(move, slot)) {
throw std::logic_error("");
}
position.make_move(slot);
}
return encoded;
}

std::optional<std::vector<Move>> rebound_goal_path(
const detail::SearchPosition &state, Player goal_owner) {
std::fill(parents_.begin(), parents_.end(), kUnseen);
const auto start = state.ball_vertex();
parents_[start] = -1;
std::size_t head = 0;
std::size_t tail = 0;
queue_[tail++] = start;
const Player mover = state.to_move();
const RulesConfig &rules = topology_->config();
const auto &root_adjacency = topology_->adjacency(start, mover);
const Point center_goal{rules.width / 2,
goal_owner == Player::One
? 0
: rules.height + 2};
for (std::uint8_t index = 0; index < root_adjacency.count; ++index) {
const auto &arc = root_adjacency.arcs[index];
if (!state.edge_used(arc.edge) &&
topology_->point(arc.destination) == center_goal) {
return std::vector<Move>{Move{center_goal}};
}
}
const unsigned symmetry = canonical_key(state).second;

while (head < tail) {
if (deadline_expired()) {
return std::nullopt;
}
const auto vertex = queue_[head++];
const auto &adjacency = topology_->adjacency(vertex, mover);
for (unsigned direction = 0; direction < 8; ++direction) {
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
const Point destination = topology_->point(arc.destination);
if (canonical_direction(topology_->point(vertex), destination,
symmetry) != direction) {
continue;
}
if (state.edge_used(arc.edge)) {
continue;
}
if (is_goal_point(rules, destination)) {
const Player winner =
is_attacking_goal(rules, destination, Player::One)
? Player::One
: Player::Two;
if (winner != goal_owner) {
continue;
}
parents_[arc.destination] = static_cast<int>(vertex);
std::vector<Move> path;
for (auto current = arc.destination; current != start;
current = static_cast<detail::SearchTopology::VertexIndex>(
parents_[current])) {
path.push_back(Move{topology_->point(current)});
}
std::reverse(path.begin(), path.end());
return path;
}
if (parents_[arc.destination] != kUnseen) {
continue;
}
const bool continues =
is_boundary_point(rules, destination) ||
state.vertex_visited(arc.destination);
if (!continues) {
continue;
}
parents_[arc.destination] = static_cast<int>(vertex);
queue_[tail++] = arc.destination;
}
}
}
return std::nullopt;
}

ReboundOutcome analyze_rebound_component(
const detail::SearchPosition &state) {
if (rebound_goal_path(state, state.to_move()).has_value()) {
return ReboundOutcome::Win;
}

std::fill(parents_.begin(), parents_.end(), kUnseen);
const auto start = state.ball_vertex();
parents_[start] = -1;
std::size_t head = 0;
std::size_t tail = 0;
queue_[tail++] = start;
const Player mover = state.to_move();
const RulesConfig &rules = topology_->config();
bool safe_handoff = false;

while (head < tail) {
if (deadline_expired()) {
return ReboundOutcome::Unknown;
}
const auto vertex = queue_[head++];
const auto &adjacency = topology_->adjacency(vertex, mover);
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (state.edge_used(arc.edge)) {
continue;
}
const Point destination = topology_->point(arc.destination);
if (is_goal_point(rules, destination) ||
parents_[arc.destination] != kUnseen) {
continue;
}
if (!is_boundary_point(rules, destination) &&
!state.vertex_visited(arc.destination)) {
const auto &reply =
topology_->adjacency(arc.destination, opponent(mover));
for (std::uint8_t child = 0; child < reply.count; ++child) {
if (reply.arcs[child].edge != arc.edge &&
!state.edge_used(reply.arcs[child].edge)) {
safe_handoff = true;
break;
}
}
continue;
}
parents_[arc.destination] = static_cast<int>(vertex);
queue_[tail++] = arc.destination;
}
}
if (!safe_handoff &&
rules.blocked_rule == BlockedRule::MoverLoses) {
return ReboundOutcome::Loss;
}
return ReboundOutcome::Unknown;
}

TacticalClass classify(const detail::SearchPosition &result) {
if (result.is_terminal()) {
return result.winner() == root_.to_move()
? TacticalClass::ImmediateWin
: TacticalClass::TerminalLoss;
}
switch (analyze_rebound_component(result)) {
case ReboundOutcome::Win:
return TacticalClass::OpponentImmediateWin;
case ReboundOutcome::Loss:
return TacticalClass::ForcedCutoff;
case ReboundOutcome::Unknown:
return TacticalClass::SafeHandoff;
}
throw std::logic_error("");
}

void retain(GenerationResult &output, std::vector<Move> moves,
detail::SearchPosition result) {
if (deadline_expired()) {
return;
}
++output.stats.completed_actions;
output.stats.max_action_edges = std::max<std::uint32_t>(
output.stats.max_action_edges,
static_cast<std::uint32_t>(moves.size()));
const detail::PositionKey key = boundary_key(result);
if (std::find(retained_keys_.begin(), retained_keys_.end(), key) !=
retained_keys_.end()) {
++output.stats.duplicates;
return;
}
retained_keys_.push_back(key);
const TacticalClass tactical = classify(result);
if (deadline_reached_) {
return;
}
switch (tactical) {
case TacticalClass::ImmediateWin:
++output.stats.immediate_wins;
++output.stats.tactical_actions;
break;
case TacticalClass::ForcedCutoff:
++output.stats.forced_cutoffs;
++output.stats.tactical_actions;
break;
case TacticalClass::SafeHandoff:
++output.stats.safe_handoffs;
break;
case TacticalClass::OpponentImmediateWin:
++output.stats.opponent_immediate_wins;
++output.stats.tactical_actions;
break;
case TacticalClass::TerminalLoss:
++output.stats.terminal_losses;
++output.stats.tactical_actions;
break;
}
const std::string encoded = encode_action(moves);
output.actions.emplace_back(std::move(moves), std::move(result), tactical,
encoded);
}

void retain_bounded_handoff_paths(GenerationResult &output) {
if (output.actions.size() >= config_.max_actions) {
return;
}
const std::uint64_t available = std::min(
config_.max_partial_paths - output.stats.explored_partial_paths,
config_.max_work - output.stats.explored_partial_paths);
const std::uint64_t limit =
std::min<std::uint64_t>(kTacticalPartialPaths, available);
std::array<std::optional<Partial>, 5> candidates;
std::deque<Partial> pending;
pending.emplace_back(root_, std::vector<Move>{});
std::uint64_t extracted = 0;
while (!pending.empty() && extracted < limit) {
if (deadline_expired()) {
return;
}
++extracted;
++output.stats.explored_partial_paths;
Partial partial = [&] {
if (output.stats.explored_partial_paths % 10U == 0U) {
++output.stats.fifo_extractions;
Partial value = std::move(pending.front());
pending.pop_front();
return value;
}
++output.stats.lifo_extractions;
Partial value = std::move(pending.back());
pending.pop_back();
return value;
}();
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = partial.position.legal_slots(slots);
shuffle(slots, count, partial.position);
for (std::uint8_t index = 0; index < count; ++index) {
Partial child = partial;
const Move move = child.position.move_for_slot(slots[index]);
child.moves.push_back(move);
child.position.make_move(slots[index]);
if (child.position.is_terminal() ||
child.position.to_move() != root_.to_move()) {
const TacticalClass tactical = classify(child.position);
if (deadline_reached_) {
return;
}
const std::size_t slot = static_cast<std::size_t>(tactical);
if (tactical != TacticalClass::ImmediateWin &&
!candidates[slot].has_value()) {
candidates[slot] = std::move(child);
}
} else if (pending.size() < config_.max_partial_paths) {
pending.push_back(std::move(child));
output.stats.max_deque_size = std::max<std::uint32_t>(
output.stats.max_deque_size,
static_cast<std::uint32_t>(pending.size()));
}
}
}
constexpr std::array<TacticalClass, 4> priority{{
TacticalClass::ForcedCutoff, TacticalClass::SafeHandoff,
TacticalClass::OpponentImmediateWin, TacticalClass::TerminalLoss}};
for (const TacticalClass tactical : priority) {
auto &candidate = candidates[static_cast<std::size_t>(tactical)];
if (candidate.has_value() &&
output.actions.size() < config_.max_actions) {
retain(output, std::move(candidate->moves),
std::move(candidate->position));
}
}
}

void retain_exact_goal_path(GenerationResult &output, Player goal_owner) {
const auto path = rebound_goal_path(root_, goal_owner);
if (deadline_reached_ || !path.has_value() ||
output.actions.size() >= config_.max_actions) {
return;
}
detail::SearchPosition result = root_;
for (const Move move : *path) {
std::uint8_t slot = 0;
if (!result.slot_for_move(move, slot)) {
throw std::logic_error("");
}
result.make_move(slot);
}
retain(output, *path, std::move(result));
}

std::vector<Move> fallback_action(
const detail::SearchPosition &state) const {
detail::SearchPosition position = state;
const Player mover = position.to_move();
std::vector<Move> action;
while (!position.is_terminal() && position.to_move() == mover) {
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = position.legal_slots(slots);
if (count == 0) {
throw std::logic_error("");
}
std::uint8_t best = slots[0];
int best_score = std::numeric_limits<int>::min();
for (std::uint8_t index = 0; index < count; ++index) {
const Move move = position.move_for_slot(slots[index]);
int score = position.grants_extra_turn(slots[index]) ? 10'000 : 0;
const int progress = mover == Player::One
? position.ball().y - move.to.y
: move.to.y - position.ball().y;
score += progress * 100;
if (is_attacking_goal(topology_->config(), move.to, mover)) {
score += 1'000'000;
} else if (is_goal_point(topology_->config(), move.to)) {
score -= 1'000'000;
}
if (score > best_score ||
(score == best_score && slots[index] < best)) {
best = slots[index];
best_score = score;
}
}
action.push_back(position.move_for_slot(best));
position.make_move(best);
}
return action;
}

};

double normalized_search_value(int value, bool proven = true) noexcept {
if (proven &&
std::abs(value) >= kMateScore - kMaximumMateDistance) {
return value > 0 ? 10'000.0 : -10'000.0;
}
return static_cast<double>(
std::clamp(value, -kMaximumEvaluation, kMaximumEvaluation)) /
kMaximumEvaluation;
}

double uct_selection_score(int value, Player mover,
std::uint32_t parent_visits,
std::uint32_t child_visits) noexcept {
const double exploitation =
player_sign(mover) * normalized_search_value(value);
if (child_visits == 0) {
return exploitation + kFirstPlayUrgency;
}
return exploitation +
kExplorationConstant *
std::sqrt(std::log(static_cast<double>(
std::max<std::uint32_t>(1, parent_visits))) /
static_cast<double>(child_visits));
}

int minimax_backup(Player mover, const std::vector<int> &values) {
if (values.empty()) {
throw std::invalid_argument("");
}
return mover == Player::One
? *std::max_element(values.begin(), values.end())
: *std::min_element(values.begin(), values.end());
}

class FullTurnBfmSearch {
public:
FullTurnBfmSearch(const GameState &state, SearchConfig config)
: config_(config),
topology_(std::make_shared<detail::SearchTopology>(state.config)),
root_position_(topology_, state), position_(root_position_),
evaluations_(config.evaluation_entries),
distances_(topology_->vertex_count(), -1),
queue_(topology_->vertex_count()),
teacher_residual_root_enabled_(
state.used_segments.size() >=
static_cast<std::size_t>(kTeacherResidualMinimumUsedEdges)) {
if (config_.max_work == 0 || config_.max_work > kMaximumWork ||
config_.max_tree_nodes < 2 ||
config_.max_tree_nodes > kMaximumTreeNodes ||
config_.max_actions == 0 || config_.max_actions > kMaximumActions ||
config_.max_partial_paths == 0 ||
config_.max_partial_paths > kMaximumPartialPaths ||
config_.evaluation_entries < 2 ||
config_.replay_value_blend_percent < 0 ||
config_.replay_value_blend_percent > 100 ||
config_.teacher_residual_weight_percent < 0 ||
config_.teacher_residual_weight_percent > 100 ||
!std::isfinite(config_.exploration_constant) ||
config_.exploration_constant < 0.0 ||
!std::isfinite(config_.first_play_urgency)) {
throw std::invalid_argument("");
}
initialize_rotation_maps();
if (config_.absolute_deadline.has_value()) {
deadline_ = config_.absolute_deadline;
} else if (config_.max_time_ms != 0) {
deadline_ =
SearchClock::now() + std::chrono::milliseconds(config_.max_time_ms);
}
nodes_.reserve(config_.max_tree_nodes);
}

std::vector<Move> run() {
if (root_position_.is_terminal()) {
throw std::invalid_argument("");
}

const std::vector<Move> fallback = fallback_action(root_position_);
if (!budget_available()) {
return fallback;
}
position_ = root_position_;
const int root_evaluation = cached_evaluate();
if (deadline_expired()) {
return fallback;
}
nodes_.emplace_back(root_position_, kNoParent, std::vector<Move>{},
std::string{}, root_evaluation, 0, false);
stats_.tree_nodes = 1;
stats_.root_score = root_evaluation;

if (!expand_node(0)) {
stats_.budget_exhausted = true;
return fallback;
}
++nodes_[0].visits;
refresh_node(0);

while (!nodes_[0].closed && budget_available()) {
const std::optional<std::vector<std::size_t>> selected = select_path();
if (!selected.has_value() || selected->empty()) {
break;
}
const std::size_t leaf = selected->back();
if (!expand_node(leaf)) {
break;
}
for (const std::size_t index : *selected) {
++nodes_[index].visits;
}
backup_path(*selected);
}

if (nodes_[0].children.empty()) {
stats_.budget_exhausted = true;
return fallback;
}
const std::size_t chosen = final_child();
stats_.root_score = nodes_[chosen].value;
return nodes_[chosen].incoming_action;
}

const SearchStats &stats() const noexcept { return stats_; }

EvaluationSnapshot evaluation_snapshot() {
position_ = root_position_;
return make_evaluation_snapshot();
}

private:
static constexpr std::size_t kNoParent =
std::numeric_limits<std::size_t>::max();

struct TreeNode {
detail::SearchPosition position;
std::size_t parent{kNoParent};
std::vector<std::size_t> children;
std::vector<Move> incoming_action;
std::string action_key;
int value{};
std::uint32_t visits{};
std::uint32_t depth{};
bool expanded{};
bool solved{};
bool exhaustive{};
bool closed{};

TreeNode(detail::SearchPosition state, std::size_t parent_index,
std::vector<Move> action, std::string key, int score,
std::uint32_t node_depth, bool proven)
: position(std::move(state)), parent(parent_index),
incoming_action(std::move(action)), action_key(std::move(key)),
value(score), depth(node_depth), expanded(proven),
solved(proven), exhaustive(proven), closed(proven) {}
};

SearchConfig config_;
std::shared_ptr<const detail::SearchTopology> topology_;
detail::SearchPosition root_position_;
detail::SearchPosition position_;
EvaluationTable evaluations_;
SearchStats stats_;
std::vector<int> distances_;
std::vector<detail::SearchTopology::VertexIndex> queue_;
std::deque<detail::SearchTopology::VertexIndex> distance_queue_;
std::vector<detail::SearchTopology::VertexIndex> rotated_vertices_;
std::vector<detail::SearchTopology::EdgeIndex> rotated_edges_;
std::optional<SearchClock::time_point> deadline_;
std::vector<TreeNode> nodes_;
bool teacher_residual_root_enabled_{};

bool deadline_expired() {
if (deadline_.has_value() && SearchClock::now() >= *deadline_) {
stats_.deadline_reached = true;
stats_.budget_exhausted = true;
return true;
}
return false;
}

bool budget_available() {
if (stats_.work >= config_.max_work) {
stats_.budget_exhausted = true;
return false;
}
if (!nodes_.empty() && nodes_.size() >= config_.max_tree_nodes) {
stats_.node_cap_reached = true;
stats_.budget_exhausted = true;
return false;
}
return !deadline_expired();
}

int terminal_score(const detail::SearchPosition &state,
std::uint32_t turn_depth) {
const std::optional<Player> winning_player = state.winner();
if (!winning_player.has_value()) {
throw std::logic_error("");
}
return proven_score(*winning_player, turn_depth);
}

static int proven_score(Player winning_player,
std::uint32_t turn_depth) noexcept {
const int distance = static_cast<int>(
std::min<std::uint32_t>(turn_depth, kMaximumMateDistance));
return winning_player == Player::One ? kMateScore - distance
: -kMateScore + distance;
}

static bool score_wins_for(int score, Player player) noexcept {
return std::abs(score) >= kMateScore - kMaximumMateDistance &&
(score > 0) == (player == Player::One);
}

void add_generator_stats(const GeneratorStats &source) {
stats_.generator_partial_paths += source.explored_partial_paths;
stats_.completed_actions += source.completed_actions;
stats_.generator_duplicates += source.duplicates;
stats_.tactical_actions += source.tactical_actions;
stats_.generator_truncations += source.truncations;
stats_.max_deque_size =
std::max(stats_.max_deque_size, source.max_deque_size);
stats_.max_action_edges =
std::max(stats_.max_action_edges, source.max_action_edges);
stats_.deadline_reached =
stats_.deadline_reached || source.deadline_reached;
}

bool expand_node(std::size_t node_index) {
if (!budget_available() || nodes_[node_index].expanded ||
nodes_[node_index].solved) {
return false;
}
const std::uint64_t remaining_work = config_.max_work - stats_.work;
GeneratorConfig generator_config;
generator_config.max_actions = config_.max_actions;
generator_config.max_partial_paths = config_.max_partial_paths;
generator_config.max_work = remaining_work;
generator_config.absolute_deadline = deadline_;
FullTurnGenerator generator(topology_, nodes_[node_index].position,
generator_config);
GenerationResult generated = generator.run();
add_generator_stats(generated.stats);
stats_.work += generated.stats.explored_partial_paths;

std::size_t expected = generated.actions.size();
if (expected == 0) {
stats_.budget_exhausted = true;
return false;
}
const bool work_fits = expected <= config_.max_work - stats_.work;
const bool tree_fits =
expected <= config_.max_tree_nodes - nodes_.size();
if (!work_fits || !tree_fits) {
const Player mover = nodes_[node_index].position.to_move();
const auto winning = std::find_if(
generated.actions.begin(), generated.actions.end(),
[mover](const CompleteTurnAction &action) {
return (action.result.is_terminal() &&
action.result.winner() == mover) ||
action.tactical == TacticalClass::ForcedCutoff;
});
if (winning != generated.actions.end() &&
stats_.work < config_.max_work &&
nodes_.size() < config_.max_tree_nodes) {
std::iter_swap(generated.actions.begin(), winning);
generated.actions.erase(generated.actions.begin() + 1,
generated.actions.end());
generated.exhaustive = false;
expected = 1;
} else {
stats_.node_cap_reached = !tree_fits;
stats_.budget_exhausted = true;
return false;
}
}

const std::size_t original_tree_size = nodes_.size();
std::size_t inserted = 0;
for (CompleteTurnAction &action : generated.actions) {
if (!budget_available()) {
break;
}
++stats_.work;
++stats_.child_evaluations;
const bool terminal = action.result.is_terminal();
const std::uint32_t child_depth = nodes_[node_index].depth + 1U;
int value = 0;
bool proven = terminal;
if (terminal) {
value = terminal_score(action.result, child_depth);
} else if (action.tactical == TacticalClass::ForcedCutoff) {
value = proven_score(opponent(action.result.to_move()),
child_depth + 1U);
proven = true;
} else if (action.tactical ==
TacticalClass::OpponentImmediateWin) {
value = proven_score(action.result.to_move(), child_depth + 1U);
proven = true;
} else {
position_ = action.result;
value = cached_evaluate();
}
const std::size_t child_index = nodes_.size();
nodes_.emplace_back(std::move(action.result), node_index,
std::move(action.moves), std::move(action.encoded),
value, child_depth, proven);
nodes_[node_index].children.push_back(child_index);
++inserted;
}

if (inserted != expected) {
nodes_.erase(nodes_.begin() +
static_cast<std::ptrdiff_t>(original_tree_size),
nodes_.end());
nodes_[node_index].children.clear();
stats_.tree_nodes = nodes_.size();
stats_.budget_exhausted = true;
return false;
}

nodes_[node_index].expanded = true;
nodes_[node_index].exhaustive = generated.exhaustive;
stats_.tree_nodes = nodes_.size();
if (nodes_[node_index].expanded) {
++stats_.expansions;
refresh_node(node_index);
return true;
}
stats_.budget_exhausted = true;
return false;
}

void refresh_node(std::size_t node_index) {
TreeNode &node = nodes_[node_index];
if (node.children.empty()) {
return;
}
std::vector<int> child_values;
child_values.reserve(node.children.size());
std::vector<int> winning_values;
winning_values.reserve(node.children.size());
bool all_solved = node.exhaustive;
bool all_closed = true;
for (const std::size_t child_index : node.children) {
const TreeNode &child = nodes_[child_index];
child_values.push_back(child.value);
all_solved = all_solved && child.solved;
all_closed = all_closed && child.closed;
if (child.solved &&
score_wins_for(child.value, node.position.to_move())) {
winning_values.push_back(child.value);
}
}
if (!winning_values.empty()) {
node.value =
minimax_backup(node.position.to_move(), winning_values);
node.solved = true;
} else {
node.value = minimax_backup(node.position.to_move(), child_values);
node.solved = all_solved;
}
node.closed = node.solved || all_closed;
}

void backup_path(const std::vector<std::size_t> &path) {
for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
refresh_node(*iterator);
}
}

double selection_score(const TreeNode &parent,
const TreeNode &child) const noexcept {
const double exploitation =
player_sign(parent.position.to_move()) *
normalized_search_value(child.value, child.solved);
if (child.visits == 0) {
return exploitation + config_.first_play_urgency;
}
return exploitation +
config_.exploration_constant *
std::sqrt(std::log(static_cast<double>(
std::max<std::uint32_t>(1,
parent.visits))) /
static_cast<double>(child.visits));
}

std::optional<std::vector<std::size_t>> select_path() const {
std::vector<std::size_t> path{0};
std::size_t current = 0;
while (nodes_[current].expanded) {
const TreeNode &parent = nodes_[current];
std::optional<std::size_t> best;
double best_score = -std::numeric_limits<double>::infinity();
for (const std::size_t child_index : parent.children) {
const TreeNode &child = nodes_[child_index];
if (child.closed) {
continue;
}
const double score = selection_score(parent, child);
if (!best.has_value() || score > best_score ||
(score == best_score &&
child.action_key < nodes_[*best].action_key)) {
best = child_index;
best_score = score;
}
}
if (!best.has_value()) {
return std::nullopt;
}
current = *best;
path.push_back(current);
}
return path;
}

std::size_t final_child() const {
const TreeNode &root = nodes_[0];
std::size_t best = root.children.front();
double best_score = -std::numeric_limits<double>::infinity();
for (const std::size_t child_index : root.children) {
const TreeNode &child = nodes_[child_index];
const double score =
player_sign(root.position.to_move()) *
normalized_search_value(child.value, child.solved) +
std::log(static_cast<double>(
std::max<std::uint32_t>(1, child.visits)));
if (score > best_score ||
(score == best_score &&
child.action_key < nodes_[best].action_key)) {
best = child_index;
best_score = score;
}
}
return best;
}

std::vector<Move> fallback_action(
const detail::SearchPosition &state) const {
detail::SearchPosition current = state;
const Player mover = current.to_move();
std::vector<Move> action;
while (!current.is_terminal() && current.to_move() == mover) {
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = current.legal_slots(slots);
if (count == 0) {
throw std::logic_error("");
}
std::uint8_t best = slots[0];
int best_score = std::numeric_limits<int>::min();
for (std::uint8_t index = 0; index < count; ++index) {
const Move move = current.move_for_slot(slots[index]);
int score = current.grants_extra_turn(slots[index]) ? 10'000 : 0;
const int progress = mover == Player::One
? current.ball().y - move.to.y
: move.to.y - current.ball().y;
score += progress * 100;
if (is_attacking_goal(topology_->config(), move.to, mover)) {
score += 1'000'000;
} else if (is_goal_point(topology_->config(), move.to)) {
score -= 1'000'000;
}
if (score > best_score ||
(score == best_score && slots[index] < best)) {
best = slots[index];
best_score = score;
}
}
action.push_back(current.move_for_slot(best));
current.make_move(best);
}
return action;
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
throw std::logic_error("");
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
throw std::logic_error("");
}
rotated_edges_[arc.edge] = *rotated_edge;
}
}
}
if (std::find(rotated_edges_.begin(), rotated_edges_.end(),
kMissingEdge) != rotated_edges_.end()) {
throw std::logic_error("");
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
throw std::logic_error("");
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
throw std::logic_error("");
}
return decoded;
}();
return weights;
}

ReplayValueOutput replay_value_output(Player mover) {
if (topology_->edge_count() != replay_value_model::kEdgeCount ||
topology_->vertex_count() != replay_value_model::kVertexCount) {
throw std::logic_error("");
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
for (std::size_t second = 0;
second < replay_value_model::kHiddenTwo; ++second) {
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
const EvaluationSnapshot snapshot = make_evaluation_snapshot();
int score = snapshot.anchor_score;
if (config_.teacher_residual_weight_percent != 0 &&
teacher_residual_root_enabled_) {
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
const detail::PositionKey key = boundary_key(position_);
if (const std::optional<int> cached = evaluations_.find(key)) {
return *cached;
}
const int score = evaluate();
evaluations_.store(key, score);
return score;
}
};

void apply_encoded_turn(GameState &state, std::string_view encoded) {
if (encoded.empty()) {
throw std::invalid_argument("");
}
GameState next = state;
const Player mover = next.to_move;
for (const char direction : encoded) {
if (is_terminal(next) || next.to_move != mover) {
throw std::invalid_argument("");
}
next = apply_move(next, decode_direction(next.ball, direction));
}
if (!is_terminal(next) && next.to_move == mover) {
throw std::invalid_argument("");
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
encoded.assign(recorded.substr(action_begin, action_end - action_begin));
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
config.replay_value_blend_percent = 15;
config.teacher_residual_weight_percent = 100;
config.absolute_deadline =
SearchClock::now() + std::chrono::milliseconds(search_time_ms);
FullTurnBfmSearch search(state, config);
const std::vector<Move> action = search.run();
if (action.empty()) {
throw std::logic_error("");
}
const Player mover = state.to_move;
std::string encoded;
encoded.reserve(action.size());
for (const Move move : action) {
if (is_terminal(state) || state.to_move != mover) {
throw std::logic_error("");
}
encoded.push_back(encode_direction(state.ball, move.to));
state = apply_move(state, move);
}
if (!is_terminal(state) && state.to_move == mover) {
throw std::logic_error("");
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
