
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
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

namespace papersoccer::jacek_native_model {

inline constexpr std::size_t kInputs = 1156;
inline constexpr std::size_t kHiddenOne = 32;
inline constexpr std::size_t kHiddenTwo = 32;
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = 38048;
inline constexpr std::size_t kPackedByteCount = 14268;
inline constexpr int kQuantizationBits = 3;
inline constexpr unsigned long long kTrainingSeed = 20260813ULL;
inline constexpr bool kBootstrapSeed = false;
inline constexpr float kScale1 = 0.525822222F;
inline constexpr float kScale2 = 0.21545966F;
inline constexpr float kScale3 = 0.0788499191F;
inline constexpr std::string_view kModelSchema = "jacek_native_model/v1";
inline constexpr std::string_view kFeatureSchema = "canonical-edges316-onehot-true-turn-distance105x8-v1";
inline constexpr std::string_view kModelSha256 = "19f954092bea404ab18ccc7aaec8b7f6627f0b459017a7f83b6d666b6bb03acc";
inline constexpr std::string_view kSchemaSha256 = "dd36c1b2800620fab1d5dc88afe95fcbb13864d581a18f01d26b3e1c3a4a6dfd";
inline constexpr std::string_view kPackedSha256 = "7125339d76ade22b0d8e3de249876927b99611372ff81396994c074522394218";
inline constexpr std::string_view kPackedWeights =
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAIAAAAAAAAAAAAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAADgAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAA4AAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAABAAAAAAAAAAAAAAABAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAIADAAAAAAAAAAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAJADAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAEAAAAAAAAAADgAAAA"
"AAAAAIADAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAA4AABAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJADAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AA4AAJADAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQ5AAI4DAAAEQAAEAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAPAAAAAAAAAA"
"AQAgAI4DAAAAAAAEAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAcAQ5AAP4DAAAEQAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAHAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAc"
"AAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAIAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAAAAAAAAAAIAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAIADAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAHAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAADgAAAAAAAAAAAAAAAAAAAcAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAPAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAABAAAAAAAAAEAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAADgAAAEAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAIAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAADgAAAE"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAA4AAIADAADgAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAAAAAAAA"
"AAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAADgAAAAAAAAABAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIADAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAADgAAAE"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAc"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAADgAAAEAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAQAgAIADAAAAAAAEAAAAAAAAAAAAAAAA"
"AADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAAAAAAAAAAc"
"AAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEAg5gAP4DAADkQAAIAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA4AAAAAAAAAAAAAAAIAAHAAAAAAAAAcAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAE"
"AQAgAA4AAAAAQAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAcAAAAAAAAAAAAAAAAAAAAAAAAAADgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEx2EkQIAnOQDk/wEDAPJECA4ESK44D4Ic"
"+vMciQIkCCLjCGIAAPLgiXEcgP8EEP4gQIPgAPAHBgDkAPDnQBzoxwEcSAHgyZ/oyZE8AQAcOHBEP4ADEJ5HgXDnQJAnCTA8"
"QHDgOIw/T4IIiSEASYEHCIDryQEHCGD/yHEfCA6gPy580B/kefL0wGEjyHEACHA8AP4DwHHbAI4fx3McOA7cwI8fwB8ceAA8"
"/4Pnf4ADMHDgP3AAAPAHhw0ECAAAOY4APwAEEJLAwIE8OIAAAJIAS4D8AB4AEYIHOBDjAPQCP4Ljh+IcwHMAue8g+PH8x4Eb"
"AAAAz/EA+QEgzxPgAJwEwIU4AR4d14PEB4BHgHLgv/TkABTk8YH/AJAHiC8YOX4cOXIgwQPfFwIAD2A8gZ8HAYTj8H8AD44H"
"T5JHCRALOA4IuJH4+Q8gyA8AAP4cAIwfT5L/wA8EORAEgHEcQBLkwI/feAAER3LjwHEARwAARvDeN/D/eHTlCfIAf4IkQBI8"
"zZtPQhDkQIQHwKMb";

}  // namespace papersoccer::jacek_native_model

namespace papersoccer::jacek_native_bfm {

using SearchClock = std::chrono::steady_clock;
using SearchTopology = detail::SearchTopology;
using SearchPosition = detail::SearchPosition;
using PositionKey = detail::PositionKey;

[[noreturn]] void invalid(const char *message) {
throw std::invalid_argument(message);
}

[[noreturn]] void impossible(const char *message) {
throw std::logic_error(message);
}

inline constexpr std::uint32_t kFirstSearchTimeMs = 800;
inline constexpr std::uint32_t kLaterSearchTimeMs = 155;
inline constexpr std::size_t kMaximumActions = 250;
inline constexpr std::size_t kMaximumPartialPaths = 50'000;
inline constexpr std::size_t kProductionPartialPaths = 4'000;
inline constexpr std::size_t kNonrootPartialPaths = 512;
inline constexpr std::size_t kMaximumTreeNodes = 120'000;
inline constexpr std::size_t kProductionTreeNodes = 80'000;
inline constexpr std::uint64_t kMaximumExpansions = 2'000'000;
inline constexpr std::size_t kFifoPeriod = 10;
inline constexpr std::size_t kLifoExtractionsPerPeriod = 9;
inline constexpr std::size_t kProofPaths = 64;
inline constexpr double kExplorationConstant = 0.95;
inline constexpr double kFirstPlayUrgency = 0.5;
inline constexpr GoalRule kProductionGoalRule = GoalRule::OwnGoalsAllowed;
inline constexpr BlockedRule kProductionBlockedRule = BlockedRule::MoverLoses;
inline constexpr std::size_t kEdgeInputs = 316;
inline constexpr std::size_t kVertices = 105;
inline constexpr std::size_t kDistanceBuckets = 8;
inline constexpr std::size_t kFeatureCount =
kEdgeInputs + kVertices * kDistanceBuckets;
inline constexpr std::size_t kFeatureInputs = kFeatureCount;
inline constexpr std::size_t kEdgeFeatureInputs = kEdgeInputs;
inline constexpr std::size_t kFeatureVertices = kVertices;
inline constexpr std::size_t kHiddenOne = 32;
inline constexpr std::size_t kHiddenTwo = 32;
inline constexpr float kMateScore = 10'000.0F;

static_assert(kFeatureCount == 1156);
static_assert(kFifoPeriod - 1 == kLifoExtractionsPerPeriod);
static_assert(jacek_native_model::kInputs == kFeatureCount);
static_assert(jacek_native_model::kHiddenOne == kHiddenOne);
static_assert(jacek_native_model::kHiddenTwo == kHiddenTwo);

inline constexpr std::array<Point, 8> kDirectionDeltas{{
{0, -1}, {1, -1}, {1, 0}, {1, 1},
{0, 1},  {-1, 1}, {-1, 0}, {-1, -1},
}};

using DenseFeatures = std::array<std::uint8_t, kFeatureCount>;
using FirstLayer = std::array<float, kHiddenOne>;

int player_sign(Player player) noexcept {
return player == Player::One ? 1 : -1;
}

bool is_production_rules(const RulesConfig &rules) noexcept {
return rules.width == 8 && rules.height == 10 &&
rules.goal_rule == kProductionGoalRule &&
rules.blocked_rule == kProductionBlockedRule;
}

Player player_for_id(int player_id) {
if (player_id == 0) return Player::One;
if (player_id == 1) return Player::Two;
invalid("player id");
}

Move decode_direction(Point from, char direction) {
if (direction < '0' || direction > '7') {
invalid("invalid direction");
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
invalid("direction move");
}

class SplitMix64 {
public:
explicit SplitMix64(std::uint64_t state) : state_(state) {}

std::uint64_t next() noexcept {
state_ += 0x9e3779b97f4a7c15ULL;
return detail::mix_position_key(state_);
}

std::size_t index(std::size_t bound) noexcept {
if (bound < 2) return 0;
const std::uint64_t threshold = static_cast<std::uint64_t>(-bound) % bound;
for (;;) {
const std::uint64_t value = next();
if (value >= threshold) return static_cast<std::size_t>(value % bound);
}
}

private:
std::uint64_t state_{};
};

float first_activation(float value) noexcept {
return value < 0.0F ? 0.01F * value : value * value;
}

float second_activation(float value) noexcept {
return value < 0.0F ? 0.01F * value : value;
}

float jacek_fast_tanh(float value) noexcept {
if (value < -4.95F) return -1.0F;
if (value > 4.95F) return 1.0F;
const float square = value * value;
const float numerator =
value * (135135.0F + square *
(17325.0F + square * (378.0F + square)));
const float denominator =
135135.0F + square *
(62370.0F + square * (3150.0F + 28.0F * square));
return numerator / denominator;
}

float terminal_value(Player winner, Player perspective,
std::uint32_t round) noexcept {
const float magnitude =
std::max(1.0F, kMateScore - 10.0F * static_cast<float>(round));
return winner == perspective ? magnitude : -magnitude;
}

bool proven_action_win(bool solved, float action_value) noexcept {
return solved && action_value > 0.0F;
}

double uct_action_score(float action_value, std::uint32_t parent_visits,
std::uint32_t child_visits,
double exploration = kExplorationConstant,
double first_play_urgency = kFirstPlayUrgency) {
if (!std::isfinite(action_value) || !std::isfinite(exploration) ||
exploration < 0.0 || !std::isfinite(first_play_urgency)) {
invalid("invalid UCT argument");
}
if (child_visits == 0) {
return static_cast<double>(action_value) + first_play_urgency;
}
return static_cast<double>(action_value) +
exploration *
std::sqrt(std::log(static_cast<double>(
std::max<std::uint32_t>(1, parent_visits))) /
static_cast<double>(child_visits));
}

double final_action_score(float action_value, std::uint32_t visits) {
if (visits == 0 || !std::isfinite(action_value)) {
invalid("unvisited final");
}
return static_cast<double>(action_value) +
std::log(static_cast<double>(visits));
}

bool valid_sha256(std::string_view value) noexcept {
if (value.size() != 64) return false;
return std::all_of(value.begin(), value.end(), [](char character) {
return (character >= '0' && character <= '9') ||
(character >= 'a' && character <= 'f');
});
}

std::uint32_t rotate_right(std::uint32_t value, unsigned shift) noexcept {
return (value >> shift) | (value << (32U - shift));
}

std::string sha256_hex(std::span<const std::uint8_t> input) {
constexpr std::array<std::uint32_t, 64> constants{{
0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
}};
std::vector<std::uint8_t> bytes(input.begin(), input.end());
const std::uint64_t bit_count = static_cast<std::uint64_t>(bytes.size()) * 8U;
bytes.push_back(0x80U);
while (bytes.size() % 64U != 56U) bytes.push_back(0U);
for (int shift = 56; shift >= 0; shift -= 8) {
bytes.push_back(static_cast<std::uint8_t>(bit_count >> shift));
}
std::array<std::uint32_t, 8> hash{{
0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
}};
for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
std::array<std::uint32_t, 64> words{};
for (std::size_t index = 0; index < 16; ++index) {
const std::size_t byte = offset + index * 4U;
words[index] = (static_cast<std::uint32_t>(bytes[byte]) << 24U) |
(static_cast<std::uint32_t>(bytes[byte + 1U]) << 16U) |
(static_cast<std::uint32_t>(bytes[byte + 2U]) << 8U) |
static_cast<std::uint32_t>(bytes[byte + 3U]);
}
for (std::size_t index = 16; index < words.size(); ++index) {
const std::uint32_t a = words[index - 15U];
const std::uint32_t b = words[index - 2U];
const std::uint32_t sigma_zero =
rotate_right(a, 7U) ^ rotate_right(a, 18U) ^ (a >> 3U);
const std::uint32_t sigma_one =
rotate_right(b, 17U) ^ rotate_right(b, 19U) ^ (b >> 10U);
words[index] = words[index - 16U] + sigma_zero +
words[index - 7U] + sigma_one;
}
std::uint32_t a = hash[0];
std::uint32_t b = hash[1];
std::uint32_t c = hash[2];
std::uint32_t d = hash[3];
std::uint32_t e = hash[4];
std::uint32_t f = hash[5];
std::uint32_t g = hash[6];
std::uint32_t h = hash[7];
for (std::size_t index = 0; index < words.size(); ++index) {
const std::uint32_t sum_one =
rotate_right(e, 6U) ^ rotate_right(e, 11U) ^ rotate_right(e, 25U);
const std::uint32_t choice = (e & f) ^ (~e & g);
const std::uint32_t temporary_one =
h + sum_one + choice + constants[index] + words[index];
const std::uint32_t sum_zero =
rotate_right(a, 2U) ^ rotate_right(a, 13U) ^ rotate_right(a, 22U);
const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
const std::uint32_t temporary_two = sum_zero + majority;
h = g;
g = f;
f = e;
e = d + temporary_one;
d = c;
c = b;
b = a;
a = temporary_one + temporary_two;
}
hash[0] += a;
hash[1] += b;
hash[2] += c;
hash[3] += d;
hash[4] += e;
hash[5] += f;
hash[6] += g;
hash[7] += h;
}
constexpr std::string_view digits = "0123456789abcdef";
std::string result(64, '0');
std::size_t position = 0;
for (const std::uint32_t word : hash) {
for (int shift = 28; shift >= 0; shift -= 4) {
result[position++] = digits[(word >> shift) & 0xfU];
}
}
return result;
}

class QuantizedModel {
public:
QuantizedModel(std::string_view packed_weights, float scale_one,
float scale_two, float scale_three,
std::string_view artifact_schema,
std::string_view feature_schema, std::string_view model_sha,
std::string_view expected_sha = {},
bool allow_empty_bootstrap = false,
std::string_view expected_packed_sha = {})
: scale_one_(scale_one), scale_two_(scale_two),
scale_three_(scale_three), model_sha_(model_sha) {
if (artifact_schema != jacek_native_model::kModelSchema ||
feature_schema != jacek_native_model::kFeatureSchema ||
!valid_sha256(model_sha) ||
(!expected_sha.empty() && model_sha != expected_sha) ||
!std::isfinite(scale_one_) || !std::isfinite(scale_two_) ||
!std::isfinite(scale_three_) || scale_one_ <= 0.0F ||
scale_two_ <= 0.0F || scale_three_ <= 0.0F) {
invalid("model metadata");
}
if (packed_weights.empty()) {
if (!allow_empty_bootstrap) {
invalid("empty payload");
}
weights_.assign(jacek_native_model::kWeightCount, 0);
packed_sha_ = sha256_hex({});
if (!expected_packed_sha.empty() &&
packed_sha_ != expected_packed_sha) {
invalid("seed hash");
}
return;
}
const std::vector<std::uint8_t> bytes = decode_base64(packed_weights);
packed_sha_ = sha256_hex(bytes);
if ((!expected_packed_sha.empty() &&
(!valid_sha256(expected_packed_sha) ||
packed_sha_ != expected_packed_sha))) {
invalid("weight hash");
}
const std::size_t expected_bytes =
(jacek_native_model::kWeightCount * 3U + 7U) / 8U;
if (bytes.size() != expected_bytes) {
invalid("weight count");
}
weights_.resize(jacek_native_model::kWeightCount);
for (std::size_t weight = 0; weight < weights_.size(); ++weight) {
std::uint8_t code = 0;
const std::size_t first_bit = weight * 3U;
for (std::size_t bit = 0; bit < 3; ++bit) {
code |= static_cast<std::uint8_t>(
(bytes[(first_bit + bit) / 8U] >>
((first_bit + bit) % 8U)) &
1U)
<< bit;
}
const int signed_value = (code & 4U) != 0U ? code - 8 : code;
if (signed_value < -3 || signed_value > 3) {
invalid("weight range");
}
weights_[weight] = static_cast<std::int8_t>(signed_value);
}
}

FirstLayer first_layer(const DenseFeatures &features) const noexcept {
FirstLayer result{};
for (std::size_t input = 0; input < features.size(); ++input) {
if (features[input] != 0) add_input(result, input, 1.0F);
}
return result;
}

void add_input(FirstLayer &layer, std::size_t input,
float multiplier) const noexcept {
const std::size_t offset = input * kHiddenOne;
for (std::size_t hidden = 0; hidden < kHiddenOne; ++hidden) {
layer[hidden] += multiplier * scale_one_ *
static_cast<float>(weights_[offset + hidden]);
}
}

float finish(FirstLayer layer) const noexcept {
for (float &value : layer) value = first_activation(value);
std::array<float, kHiddenTwo> second{};
const std::size_t second_offset = kFeatureCount * kHiddenOne;
for (std::size_t first = 0; first < kHiddenOne; ++first) {
for (std::size_t hidden = 0; hidden < kHiddenTwo; ++hidden) {
second[hidden] +=
layer[first] * scale_two_ *
static_cast<float>(weights_[second_offset +
first * kHiddenTwo + hidden]);
}
}
for (float &value : second) value = second_activation(value);
const std::size_t output_offset = second_offset + kHiddenOne * kHiddenTwo;
float output = 0.0F;
for (std::size_t hidden = 0; hidden < kHiddenTwo; ++hidden) {
output += second[hidden] * scale_three_ *
static_cast<float>(weights_[output_offset + hidden]);
}
return jacek_fast_tanh(output);
}

float evaluate(const DenseFeatures &features) const noexcept {
return finish(first_layer(features));
}

std::string_view model_sha256() const noexcept { return model_sha_; }
std::string_view packed_sha256() const noexcept { return packed_sha_; }
std::span<const std::int8_t> weights() const noexcept { return weights_; }
float scale_one() const noexcept { return scale_one_; }
float scale_two() const noexcept { return scale_two_; }
float scale_three() const noexcept { return scale_three_; }

private:
std::vector<std::int8_t> weights_;
float scale_one_{};
float scale_two_{};
float scale_three_{};
std::string model_sha_;
std::string packed_sha_;

static int base64_value(char character) noexcept {
if (character >= 'A' && character <= 'Z') return character - 'A';
if (character >= 'a' && character <= 'z') return character - 'a' + 26;
if (character >= '0' && character <= '9') return character - '0' + 52;
if (character == '+') return 62;
if (character == '/') return 63;
return -1;
}

static std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
if (encoded.size() % 4U != 0U) {
invalid("base64");
}
std::vector<std::uint8_t> result;
result.reserve(encoded.size() / 4U * 3U);
std::uint32_t buffer = 0;
int bits = 0;
bool padding = false;
for (const char character : encoded) {
if (character == '=') {
padding = true;
continue;
}
if (padding) invalid("invalid base64 padding");
const int value = base64_value(character);
if (value < 0) invalid("base64");
buffer = (buffer << 6U) | static_cast<std::uint32_t>(value);
bits += 6;
if (bits >= 8) {
bits -= 8;
result.push_back(static_cast<std::uint8_t>((buffer >> bits) & 0xffU));
}
}
return result;
}
};

const QuantizedModel &default_model() {
static const QuantizedModel model(
jacek_native_model::kPackedWeights, jacek_native_model::kScale1,
jacek_native_model::kScale2, jacek_native_model::kScale3,
jacek_native_model::kModelSchema, jacek_native_model::kFeatureSchema,
jacek_native_model::kModelSha256, jacek_native_model::kModelSha256,
jacek_native_model::kPackedWeights.empty(),
jacek_native_model::kPackedSha256);
return model;
}

struct FloatModelView {
std::span<const float> first;
std::span<const float> second;
std::span<const float> output;
};

float evaluate_float_model(const DenseFeatures &features,
FloatModelView model) {
if (model.first.size() != kFeatureCount * kHiddenOne ||
model.second.size() != kHiddenOne * kHiddenTwo ||
model.output.size() != kHiddenTwo) {
invalid("float dimensions");
}
FirstLayer first{};
for (std::size_t input = 0; input < features.size(); ++input) {
if (features[input] == 0) continue;
for (std::size_t hidden = 0; hidden < kHiddenOne; ++hidden) {
first[hidden] += model.first[input * kHiddenOne + hidden];
}
}
for (float &value : first) value = first_activation(value);
std::array<float, kHiddenTwo> second{};
for (std::size_t input = 0; input < kHiddenOne; ++input) {
for (std::size_t hidden = 0; hidden < kHiddenTwo; ++hidden) {
second[hidden] += first[input] * model.second[input * kHiddenTwo + hidden];
}
}
for (float &value : second) value = second_activation(value);
float output = 0.0F;
for (std::size_t hidden = 0; hidden < kHiddenTwo; ++hidden) {
output += second[hidden] * model.output[hidden];
}
return jacek_fast_tanh(output);
}

struct FeatureFrame {
std::array<std::uint8_t, kEdgeInputs> used_edges{};
std::array<std::uint8_t, kVertices> turn_distances{};
};

class FeatureEncoder {
public:
explicit FeatureEncoder(std::shared_ptr<const SearchTopology> topology)
: topology_(std::move(topology)),
cooperative_adjacency_(topology_->vertex_count()) {
if (topology_->config().width != 8 || topology_->config().height != 10 ||
topology_->edge_count() != kEdgeInputs ||
topology_->vertex_count() != kVertices) {
invalid("8x10 required");
}
initialize_maps();
}

FeatureFrame frame(const SearchPosition &position,
Player perspective) {
calculate_turn_distances(position);
FeatureFrame result;
const bool rotate = perspective == Player::Two;
for (std::size_t edge = 0; edge < kEdgeInputs; ++edge) {
const auto physical = rotate ? rotated_edges_[edge]
: static_cast<SearchTopology::EdgeIndex>(edge);
result.used_edges[edge] = position.edge_used(physical) ? 1 : 0;
}
for (std::size_t vertex = 0; vertex < kVertices; ++vertex) {
const auto physical =
rotate ? rotated_vertices_[vertex]
: static_cast<SearchTopology::VertexIndex>(vertex);
result.turn_distances[vertex] = static_cast<std::uint8_t>(
std::min(distances_[physical], static_cast<int>(kDistanceBuckets - 1)));
}
return result;
}

DenseFeatures dense(const SearchPosition &position, Player perspective) {
return dense(frame(position, perspective));
}

static DenseFeatures dense(const FeatureFrame &frame) noexcept {
DenseFeatures result{};
for (std::size_t edge = 0; edge < kEdgeInputs; ++edge) {
result[edge] = frame.used_edges[edge];
}
for (std::size_t vertex = 0; vertex < kVertices; ++vertex) {
const std::size_t index =
kEdgeInputs + vertex * kDistanceBuckets + frame.turn_distances[vertex];
result[index] = 1;
}
return result;
}

private:
std::shared_ptr<const SearchTopology> topology_;
std::vector<SearchTopology::Adjacency> cooperative_adjacency_;
std::array<SearchTopology::VertexIndex, kVertices> rotated_vertices_{};
std::array<SearchTopology::EdgeIndex, kEdgeInputs> rotated_edges_{};
std::array<int, kVertices> distances_{};
std::deque<SearchTopology::VertexIndex> queue_;

Point rotate(Point point) const noexcept {
return {topology_->config().width - point.x,
topology_->config().height + 2 - point.y};
}

static bool contains(const SearchTopology::Adjacency &adjacency,
const SearchTopology::Arc &candidate) noexcept {
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (arc.edge == candidate.edge && arc.destination == candidate.destination) {
return true;
}
}
return false;
}

void initialize_maps() {
std::array<std::optional<Segment>, kEdgeInputs> segments{};
for (SearchTopology::VertexIndex vertex = 0;
vertex < topology_->vertex_count(); ++vertex) {
auto &cooperative = cooperative_adjacency_[vertex];
for (const Player player : {Player::One, Player::Two}) {
const auto &source = topology_->adjacency(vertex, player);
for (std::uint8_t index = 0; index < source.count; ++index) {
const auto &arc = source.arcs[index];
if (!contains(cooperative, arc)) {
if (cooperative.count >= cooperative.arcs.size()) {
impossible("too many arcs");
}
cooperative.arcs[cooperative.count++] = arc;
}
if (!segments[arc.edge].has_value()) {
segments[arc.edge] =
Segment{topology_->point(vertex), topology_->point(arc.destination)};
}
}
}
}
for (SearchTopology::VertexIndex vertex = 0;
vertex < topology_->vertex_count(); ++vertex) {
const auto rotated = topology_->find_vertex(rotate(topology_->point(vertex)));
if (!rotated.has_value()) impossible("vertex rotation");
rotated_vertices_[vertex] = *rotated;
}
for (SearchTopology::EdgeIndex edge = 0; edge < topology_->edge_count();
++edge) {
if (!segments[edge].has_value()) impossible("missing edge");
const Segment segment = *segments[edge];
const auto rotated =
topology_->find_edge(Segment{rotate(segment.a), rotate(segment.b)});
if (!rotated.has_value()) impossible("edge rotation");
rotated_edges_[edge] = *rotated;
}
}

void calculate_turn_distances(const SearchPosition &position) {
constexpr int kUnreachable = 1'000'000;
distances_.fill(kUnreachable);
queue_.clear();
distances_[position.ball_vertex()] = 0;
queue_.push_front(position.ball_vertex());
while (!queue_.empty()) {
const auto vertex = queue_.front();
queue_.pop_front();
const auto &adjacency = cooperative_adjacency_[vertex];
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (position.edge_used(arc.edge)) continue;
const Point destination = topology_->point(arc.destination);
const bool rebounds = position.vertex_visited(arc.destination) ||
is_boundary_point(topology_->config(), destination) ||
is_goal_point(topology_->config(), destination);
const int candidate = distances_[vertex] + (rebounds ? 0 : 1);
if (candidate >= distances_[arc.destination]) continue;
distances_[arc.destination] = candidate;
if (rebounds) queue_.push_front(arc.destination);
else queue_.push_back(arc.destination);
}
}
}
};

DenseFeatures encode_features(const GameState &state) {
auto topology = std::make_shared<SearchTopology>(state.config);
SearchPosition position(topology, state);
FeatureEncoder encoder(topology);
return encoder.dense(position, position.to_move());
}

struct PreparedEvaluation {
FeatureFrame frame{};
FirstLayer first{};
Player perspective{Player::One};
};

class NeuralEvaluator {
public:
NeuralEvaluator(std::shared_ptr<const SearchTopology> topology,
const QuantizedModel &model)
: encoder_(std::move(topology)), model_(model) {}

PreparedEvaluation prepare(const SearchPosition &position,
Player perspective) {
PreparedEvaluation result;
result.frame = encoder_.frame(position, perspective);
result.first = model_.first_layer(FeatureEncoder::dense(result.frame));
result.perspective = perspective;
return result;
}

float evaluate(const SearchPosition &position) {
return model_.evaluate(encoder_.dense(position, position.to_move()));
}

float evaluate_child(const PreparedEvaluation &base,
const SearchPosition &child) {
if (child.to_move() != base.perspective) {
invalid("evaluation perspective");
}
const FeatureFrame next = encoder_.frame(child, base.perspective);
FirstLayer first = base.first;
for (std::size_t edge = 0; edge < kEdgeInputs; ++edge) {
const int delta = static_cast<int>(next.used_edges[edge]) -
static_cast<int>(base.frame.used_edges[edge]);
if (delta != 0) model_.add_input(first, edge, static_cast<float>(delta));
}
for (std::size_t vertex = 0; vertex < kVertices; ++vertex) {
const std::uint8_t before = base.frame.turn_distances[vertex];
const std::uint8_t after = next.turn_distances[vertex];
if (before == after) continue;
model_.add_input(first,
kEdgeInputs + vertex * kDistanceBuckets + before, -1.0F);
model_.add_input(first,
kEdgeInputs + vertex * kDistanceBuckets + after, 1.0F);
}
return model_.finish(first);
}

private:
FeatureEncoder encoder_;
const QuantizedModel &model_;
};

enum class TacticalClass {
ImmediateGoal,
ForcedCutoff,
SafeHandoff,
OpponentImmediateGoal,
OwnGoal,
};

struct TurnTactics {
bool can_score{};
bool can_handoff{};
};

class TacticalAnalyzer {
public:
explicit TacticalAnalyzer(std::shared_ptr<const SearchTopology> topology)
: topology_(std::move(topology)) {}

TurnTactics analyze(const SearchPosition &position) const {
if (position.is_terminal()) return {};
const Player mover = position.to_move();
std::array<bool, kVertices> seen{};
std::array<SearchTopology::VertexIndex, kVertices> queue{};
std::size_t head = 0;
std::size_t tail = 0;
queue[tail++] = position.ball_vertex();
seen[position.ball_vertex()] = true;
TurnTactics result;
while (head < tail) {
const auto vertex = queue[head++];
const auto &adjacency = topology_->adjacency(vertex, mover);
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (position.edge_used(arc.edge)) continue;
const Point destination = topology_->point(arc.destination);
if (is_goal_point(topology_->config(), destination)) {
if (is_attacking_goal(topology_->config(), destination, mover)) {
result.can_score = true;
}
continue;
}
const bool rebounds = position.vertex_visited(arc.destination) ||
is_boundary_point(topology_->config(), destination);
if (!rebounds) {
const auto &replies =
topology_->adjacency(arc.destination, opponent(mover));
for (std::uint8_t reply = 0; reply < replies.count; ++reply) {
if (replies.arcs[reply].edge != arc.edge &&
!position.edge_used(replies.arcs[reply].edge)) {
result.can_handoff = true;
break;
}
}
continue;
}
if (!seen[arc.destination]) {
seen[arc.destination] = true;
queue[tail++] = arc.destination;
}
}
}
return result;
}

std::optional<std::vector<Move>> goal_path(const SearchPosition &position,
Player goal_owner) const {
if (position.is_terminal()) return std::nullopt;
constexpr int kUnseen = -2;
std::array<int, kVertices> parent{};
parent.fill(kUnseen);
std::array<SearchTopology::VertexIndex, kVertices> queue{};
std::size_t head = 0;
std::size_t tail = 0;
const auto start = position.ball_vertex();
parent[start] = -1;
queue[tail++] = start;
while (head < tail) {
const auto vertex = queue[head++];
const auto &adjacency = topology_->adjacency(vertex, position.to_move());
for (std::uint8_t index = 0; index < adjacency.count; ++index) {
const auto &arc = adjacency.arcs[index];
if (position.edge_used(arc.edge)) continue;
const Point destination = topology_->point(arc.destination);
if (is_goal_point(topology_->config(), destination)) {
const Player owner =
is_attacking_goal(topology_->config(), destination, Player::One)
? Player::One
: Player::Two;
if (owner != goal_owner) continue;
parent[arc.destination] = static_cast<int>(vertex);
std::vector<Move> path;
for (auto current = arc.destination; current != start;
current = static_cast<SearchTopology::VertexIndex>(parent[current])) {
path.push_back(Move{topology_->point(current)});
}
std::reverse(path.begin(), path.end());
return path;
}
const bool rebounds = position.vertex_visited(arc.destination) ||
is_boundary_point(topology_->config(), destination);
if (rebounds && parent[arc.destination] == kUnseen) {
parent[arc.destination] = static_cast<int>(vertex);
queue[tail++] = arc.destination;
}
}
}
return std::nullopt;
}

private:
std::shared_ptr<const SearchTopology> topology_;
};

struct GeneratorConfig {
std::size_t max_actions{kMaximumActions};
std::size_t max_partial_paths{kMaximumPartialPaths};
std::optional<SearchClock::time_point> absolute_deadline{};
std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
};

struct GeneratorStats {
std::uint64_t explored_partial_paths{};
std::uint64_t proof_paths{};
std::uint64_t completed_actions{};
std::uint64_t duplicate_boundaries{};
std::uint64_t fifo_extractions{};
std::uint64_t lifo_extractions{};
std::uint64_t tactical_actions{};
std::uint64_t truncations{};
std::uint32_t max_deque_size{};
std::uint8_t proof_classes{};
bool deadline_reached{};
bool proof_truncated{};
};

struct CompleteTurnAction {
std::vector<Move> moves;
SearchPosition result;
TacticalClass tactical{TacticalClass::SafeHandoff};
std::string encoded;
};

struct GenerationResult {
std::vector<CompleteTurnAction> actions;
GeneratorStats stats{};
bool exhaustive{};
};

class CompleteTurnGenerator {
public:
CompleteTurnGenerator(std::shared_ptr<const SearchTopology> topology,
const SearchPosition &root, GeneratorConfig config = {})
: topology_(std::move(topology)), root_(root), config_(config),
tactics_(topology_) {
if (config_.max_actions == 0 || config_.max_actions > kMaximumActions ||
config_.max_partial_paths == 0 ||
config_.max_partial_paths > kMaximumPartialPaths) {
invalid("generator limits");
}
}

GenerationResult run() {
if (root_.is_terminal()) {
invalid("terminal generation");
}
GenerationResult output;
output.actions.reserve(config_.max_actions);
retain_goal_path(output, root_.to_move());
retain_goal_path(output, opponent(root_.to_move()));
retain_witnesses(output);

std::deque<Partial> pending;
pending.push_back(Partial{root_, {}});
output.stats.max_deque_size =
std::max<std::uint32_t>(output.stats.max_deque_size, 1);
bool truncated = false;
while (!pending.empty()) {
if (output.stats.explored_partial_paths >= config_.max_partial_paths) {
truncated = true;
break;
}
if (deadline_expired()) {
output.stats.deadline_reached = true;
truncated = true;
break;
}
++output.stats.explored_partial_paths;
Partial partial = [&]() {
if (output.stats.explored_partial_paths % kFifoPeriod == 0) {
++output.stats.fifo_extractions;
Partial result = std::move(pending.front());
pending.pop_front();
return result;
}
++output.stats.lifo_extractions;
Partial result = std::move(pending.back());
pending.pop_back();
return result;
}();
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = partial.position.legal_slots(slots);
shuffle(slots, count, partial.position);
for (std::uint8_t index = 0; index < count; ++index) {
Partial child = partial;
child.moves.push_back(child.position.move_for_slot(slots[index]));
child.position.make_move(slots[index]);
if (child.position.is_terminal() ||
child.position.to_move() != root_.to_move()) {
retain(output, std::move(child));
} else if (pending.size() < config_.max_partial_paths) {
pending.push_back(std::move(child));
output.stats.max_deque_size = std::max<std::uint32_t>(
output.stats.max_deque_size,
static_cast<std::uint32_t>(pending.size()));
} else {
truncated = true;
}
}
}
if (output.actions.empty()) {
Partial fallback{root_, deterministic_legal_turn(root_)};
fallback.position = root_;
for (const Move move : fallback.moves) {
std::uint8_t slot = 0;
if (!fallback.position.slot_for_move(move, slot)) {
impossible("illegal fallback");
}
fallback.position.make_move(slot);
}
retain(output, std::move(fallback));
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
truncated = truncated || cap_discarded_;
output.exhaustive = pending.empty() && !truncated;
if (truncated) ++output.stats.truncations;
return output;
}

private:
struct Partial {
SearchPosition position;
std::vector<Move> moves;
};

std::shared_ptr<const SearchTopology> topology_;
SearchPosition root_;
GeneratorConfig config_;
TacticalAnalyzer tactics_;
std::vector<PositionKey> retained_;
bool deadline_reached_{};
bool cap_discarded_{};

bool deadline_expired() {
if (!deadline_reached_ && config_.absolute_deadline.has_value() &&
SearchClock::now() >= *config_.absolute_deadline) {
deadline_reached_ = true;
}
return deadline_reached_;
}

void shuffle(std::array<std::uint8_t, detail::kMaximumMoves> &slots,
std::uint8_t count, const SearchPosition &position) const {
const PositionKey key = position.position_key();
SplitMix64 random(config_.shuffle_seed ^ key.first ^
detail::mix_position_key(key.second));
for (std::size_t remaining = count; remaining > 1; --remaining) {
std::swap(slots[remaining - 1], slots[random.index(remaining)]);
}
}

std::string encode(const std::vector<Move> &moves) const {
SearchPosition position = root_;
std::string result;
result.reserve(moves.size());
for (const Move move : moves) {
result.push_back(encode_direction(position.ball(), move.to));
std::uint8_t slot = 0;
if (!position.slot_for_move(move, slot)) {
impossible("illegal action");
}
position.make_move(slot);
}
return result;
}

TacticalClass classify(const SearchPosition &result) const {
if (result.is_terminal()) {
return result.winner() == root_.to_move() ? TacticalClass::ImmediateGoal
: TacticalClass::OwnGoal;
}
const TurnTactics next = tactics_.analyze(result);
if (next.can_score) return TacticalClass::OpponentImmediateGoal;
if (!next.can_handoff) return TacticalClass::ForcedCutoff;
return TacticalClass::SafeHandoff;
}

void retain(GenerationResult &output, Partial partial) {
++output.stats.completed_actions;
const PositionKey key = partial.position.position_key();
for (std::size_t index = 0; index < retained_.size(); ++index) {
if (retained_[index] != key ||
!output.actions[index].result.same_compact_state(partial.position)) {
continue;
}
++output.stats.duplicate_boundaries;
const TacticalClass tactical = output.actions[index].tactical;
if (tactical != TacticalClass::SafeHandoff) {
++output.stats.tactical_actions;
}
std::string encoded = encode(partial.moves);
if (encoded < output.actions[index].encoded) {
output.actions[index] = CompleteTurnAction{
std::move(partial.moves), std::move(partial.position), tactical,
std::move(encoded)};
}
return;
}
const TacticalClass tactical = classify(partial.position);
if (tactical != TacticalClass::SafeHandoff) {
++output.stats.tactical_actions;
}
CompleteTurnAction candidate{
std::move(partial.moves), std::move(partial.position), tactical, {}};
candidate.encoded = encode(candidate.moves);
if (output.actions.size() < config_.max_actions) {
retained_.push_back(key);
output.actions.push_back(std::move(candidate));
return;
}
const auto worse = [](const CompleteTurnAction &left,
const CompleteTurnAction &right) {
if (left.tactical != right.tactical) {
return static_cast<int>(left.tactical) >
static_cast<int>(right.tactical);
}
return left.encoded > right.encoded;
};
std::size_t worst = 0;
for (std::size_t index = 1; index < output.actions.size(); ++index) {
if (worse(output.actions[index], output.actions[worst])) worst = index;
}
if (worse(output.actions[worst], candidate)) {
retained_[worst] = key;
output.actions[worst] = std::move(candidate);
}
cap_discarded_ = true;
}

void retain_goal_path(GenerationResult &output, Player goal_owner) {
if (deadline_expired()) return;
const auto path = tactics_.goal_path(root_, goal_owner);
if (!path.has_value()) return;
Partial partial{root_, *path};
for (const Move move : partial.moves) {
std::uint8_t slot = 0;
if (!partial.position.slot_for_move(move, slot)) {
impossible("illegal goal path");
}
partial.position.make_move(slot);
}
retain(output, std::move(partial));
}

void retain_witnesses(GenerationResult &output) {
constexpr std::size_t limit = kProofPaths;
std::array<std::optional<Partial>, 3> best{};
std::deque<Partial> pending;
pending.push_back(Partial{root_, {}});
while (!pending.empty() && output.stats.proof_paths < limit) {
if (deadline_expired()) {
output.stats.deadline_reached = true;
output.stats.proof_truncated = true;
break;
}
++output.stats.proof_paths;
Partial partial = std::move(pending.front());
pending.pop_front();
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = partial.position.legal_slots(slots);
shuffle(slots, count, partial.position);
for (std::uint8_t index = 0; index < count; ++index) {
Partial child = partial;
child.moves.push_back(child.position.move_for_slot(slots[index]));
child.position.make_move(slots[index]);
if (child.position.is_terminal() ||
child.position.to_move() != root_.to_move()) {
const int witness = static_cast<int>(classify(child.position)) -
static_cast<int>(TacticalClass::ForcedCutoff);
if (witness < 0 || witness >= static_cast<int>(best.size())) continue;
std::string encoded = encode(child.moves);
auto &retained = best[static_cast<std::size_t>(witness)];
if (!retained.has_value() || encoded < encode(retained->moves)) {
retained = std::move(child);
}
} else if (pending.size() < limit) {
pending.push_back(std::move(child));
output.stats.max_deque_size = std::max<std::uint32_t>(
output.stats.max_deque_size,
static_cast<std::uint32_t>(pending.size()));
} else {
output.stats.proof_truncated = true;
}
}
}
if (!pending.empty()) output.stats.proof_truncated = true;
for (auto &witness : best) {
if (!witness.has_value()) continue;
++output.stats.proof_classes;
retain(output, std::move(*witness));
}
}

std::vector<Move> deterministic_legal_turn(const SearchPosition &root) const {
SearchPosition position = root;
const Player mover = root.to_move();
std::vector<Move> result;
while (!position.is_terminal() && position.to_move() == mover) {
std::array<std::uint8_t, detail::kMaximumMoves> slots{};
const std::uint8_t count = position.legal_slots(slots);
if (count == 0) impossible("missing move");
shuffle(slots, count, position);
result.push_back(position.move_for_slot(slots[0]));
position.make_move(slots[0]);
}
return result;
}
};

struct SearchConfig {
std::size_t max_actions{kMaximumActions};
std::size_t max_partial_paths{kProductionPartialPaths};
std::size_t max_tree_nodes{kProductionTreeNodes};
std::uint64_t max_expansions{kMaximumExpansions};
std::optional<SearchClock::time_point> absolute_deadline{};
std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
double exploration_constant{kExplorationConstant};
double first_play_urgency{kFirstPlayUrgency};
const QuantizedModel *model{};
};

struct SearchStats {
std::uint64_t expansions{};
std::uint64_t generated_children{};
std::uint64_t child_evaluations{};
std::uint64_t tactical_child_values{};
std::uint64_t completed_actions{};
std::uint64_t duplicate_boundaries{};
std::uint64_t generator_partial_paths{};
std::uint64_t proof_paths{};
std::uint64_t generator_fifo_extractions{};
std::uint64_t generator_lifo_extractions{};
std::uint64_t tactical_actions{};
std::uint64_t proof_classes{};
std::uint64_t proof_truncations{};
std::uint64_t generator_truncations{};
std::uint64_t root_generator_truncations{};
std::uint64_t nonroot_generator_truncations{};
std::size_t tree_nodes{};
std::uint32_t max_complete_turn_depth{};
bool deadline_reached{};
bool tree_cap_reached{};
bool expansion_cap_reached{};
};

struct RootActionStat {
std::string encoded;
float value{};
float initial_value{};
std::uint32_t visits{};
std::uint32_t selection_visits{};
TacticalClass tactical{TacticalClass::SafeHandoff};
};

struct SearchResult {
std::vector<Move> moves;
std::string encoded;
float value{};
bool solved{};
std::optional<Player> solved_winner{};
std::vector<RootActionStat> root_actions;
SearchStats stats{};
};

class BestFirstMinimaxSearch {
public:
BestFirstMinimaxSearch(const GameState &state, SearchConfig config = {})
: config_(config),
topology_(std::make_shared<SearchTopology>(state.config)),
root_(topology_, state), evaluator_(topology_, selected_model()) {
if (!is_production_rules(state.config) ||
config_.max_actions == 0 || config_.max_actions > kMaximumActions ||
config_.max_partial_paths == 0 ||
config_.max_partial_paths > kMaximumPartialPaths ||
config_.max_tree_nodes < 2 ||
config_.max_tree_nodes > kMaximumTreeNodes ||
config_.max_expansions == 0 ||
config_.max_expansions > kMaximumExpansions ||
!std::isfinite(config_.exploration_constant) ||
config_.exploration_constant < 0.0 ||
!std::isfinite(config_.first_play_urgency)) {
invalid("search config");
}
nodes_.reserve(config_.max_tree_nodes);
}

SearchResult run() {
if (root_.is_terminal()) invalid("terminal search root");
nodes_.emplace_back(root_, no_parent(), std::vector<Move>{},
TacticalClass::SafeHandoff, root_.to_move(), 0.0F, 0,
false, false, false);
if (!expand(0)) impossible("root expansion");
refresh(0);
while (!nodes_[0].closed && budget_available()) {
const auto path = select_path();
if (!path.has_value()) break;
for (const std::size_t index : *path) {
++nodes_[index].visits;
++nodes_[index].selection_visits;
}
const std::size_t leaf = path->back();
if (!expand(leaf)) break;
backup(*path);
}
return make_result();
}

const SearchStats &stats() const noexcept { return stats_; }

private:
struct Node {
SearchPosition position;
std::size_t parent{};
std::vector<std::size_t> children;
std::vector<Move> incoming_action;
TacticalClass tactical{TacticalClass::SafeHandoff};
Player perspective{Player::One};
float value{};
float prior_value{};
std::uint32_t visits{1};
std::uint32_t selection_visits{};
std::uint32_t depth{};
bool expanded{};
bool solved{};
bool exhaustive{};
bool closed{};
std::string action_key;

Node(SearchPosition state, std::size_t parent_index,
std::vector<Move> action, TacticalClass classification,
Player value_perspective, float node_value,
std::uint32_t node_depth, bool is_expanded, bool is_solved,
bool is_exhaustive, std::string key = {})
: position(std::move(state)), parent(parent_index),
incoming_action(std::move(action)), tactical(classification),
perspective(value_perspective), value(node_value),
prior_value(node_value), depth(node_depth),
expanded(is_expanded), solved(is_solved), exhaustive(is_exhaustive),
closed(is_solved), action_key(std::move(key)) {}
};

SearchConfig config_;
std::shared_ptr<const SearchTopology> topology_;
SearchPosition root_;
NeuralEvaluator evaluator_;
std::vector<Node> nodes_;
SearchStats stats_{};

static constexpr std::size_t no_parent() noexcept {
return std::numeric_limits<std::size_t>::max();
}

const QuantizedModel &selected_model() const {
return config_.model == nullptr ? default_model() : *config_.model;
}

bool deadline_expired() {
if (config_.absolute_deadline.has_value() &&
SearchClock::now() >= *config_.absolute_deadline) {
stats_.deadline_reached = true;
return true;
}
return false;
}

bool budget_available() {
if (stats_.expansions >= config_.max_expansions) {
stats_.expansion_cap_reached = true;
return false;
}
if (nodes_.size() >= config_.max_tree_nodes) {
stats_.tree_cap_reached = true;
return false;
}
return !deadline_expired();
}

void merge_generator_stats(const GeneratorStats &source, bool root) {
stats_.completed_actions += source.completed_actions;
stats_.duplicate_boundaries += source.duplicate_boundaries;
stats_.generator_partial_paths += source.explored_partial_paths;
stats_.proof_paths += source.proof_paths;
stats_.generator_fifo_extractions += source.fifo_extractions;
stats_.generator_lifo_extractions += source.lifo_extractions;
stats_.tactical_actions += source.tactical_actions;
stats_.proof_classes += source.proof_classes;
stats_.proof_truncations += source.proof_truncated ? 1U : 0U;
stats_.generator_truncations += source.truncations;
(root ? stats_.root_generator_truncations
: stats_.nonroot_generator_truncations) += source.truncations;
stats_.deadline_reached = stats_.deadline_reached || source.deadline_reached;
}

bool expand(std::size_t index) {
if (nodes_[index].expanded || nodes_[index].solved) return true;
if (stats_.expansions >= config_.max_expansions ||
nodes_.size() >= config_.max_tree_nodes) return false;
GeneratorConfig generator_config;
generator_config.max_actions = std::min(
config_.max_actions, config_.max_tree_nodes - nodes_.size());
generator_config.max_partial_paths =
index == 0 ? config_.max_partial_paths
: std::min(config_.max_partial_paths,
kNonrootPartialPaths);
generator_config.absolute_deadline = config_.absolute_deadline;
generator_config.shuffle_seed =
config_.shuffle_seed ^ detail::mix_position_key(index + 1U);
CompleteTurnGenerator generator(topology_, nodes_[index].position,
generator_config);
GenerationResult generated = generator.run();
merge_generator_stats(generated.stats, index == 0);
if (generated.actions.empty()) return false;
if (generated.actions.size() > config_.max_tree_nodes - nodes_.size()) {
stats_.tree_cap_reached = true;
return false;
}

const Player child_perspective = opponent(nodes_[index].perspective);
std::optional<PreparedEvaluation> prepared;
if (!nodes_[index].position.is_terminal()) {
prepared = evaluator_.prepare(nodes_[index].position, child_perspective);
}
const std::uint32_t child_depth = nodes_[index].depth + 1U;
stats_.max_complete_turn_depth =
std::max(stats_.max_complete_turn_depth, child_depth);
const std::size_t first_child = nodes_.size();
for (CompleteTurnAction &action : generated.actions) {
float value = 0.0F;
bool solved = false;
if (action.result.is_terminal()) {
value = terminal_value(*action.result.winner(), child_perspective,
child_depth);
solved = true;
} else if (action.tactical == TacticalClass::ForcedCutoff) {
value = terminal_value(opponent(child_perspective), child_perspective,
child_depth + 1U);
solved = true;
} else if (action.tactical == TacticalClass::OpponentImmediateGoal) {
value = terminal_value(child_perspective, child_perspective,
child_depth + 1U);
solved = true;
} else {
value = evaluator_.evaluate_child(*prepared, action.result);
++stats_.child_evaluations;
}
nodes_.emplace_back(std::move(action.result), index,
std::move(action.moves), action.tactical,
child_perspective, value, child_depth, solved, solved,
true, std::move(action.encoded));
++stats_.generated_children;
if (solved) ++stats_.tactical_child_values;
}
nodes_[index].children.reserve(generated.actions.size());
for (std::size_t child = first_child; child < nodes_.size(); ++child) {
nodes_[index].children.push_back(child);
}
nodes_[index].expanded = true;
nodes_[index].exhaustive = generated.exhaustive;
++stats_.expansions;
stats_.tree_nodes = nodes_.size();
refresh(index);
return true;
}

void refresh(std::size_t index) {
Node &node = nodes_[index];
if (node.children.empty()) return;
float best = -std::numeric_limits<float>::infinity();
bool proven_win = false;
bool all_solved = node.exhaustive;
bool all_closed = true;
for (const std::size_t child_index : node.children) {
const Node &child = nodes_[child_index];
const float action_value = -child.value;
best = std::max(best, action_value);
proven_win = proven_win || proven_action_win(child.solved, action_value);
all_solved = all_solved && child.solved;
all_closed = all_closed && child.closed;
}
if (!node.exhaustive) best = std::max(best, node.prior_value);
node.value = best;
node.solved = proven_win || all_solved;
node.closed = node.solved || all_closed;
}

double selection_score(const Node &parent, const Node &child) const {
return uct_action_score(-child.value, parent.selection_visits,
child.selection_visits,
config_.exploration_constant,
config_.first_play_urgency);
}

std::optional<std::vector<std::size_t>> select_path() const {
std::vector<std::size_t> path{0};
std::size_t current = 0;
while (nodes_[current].expanded) {
const Node &parent = nodes_[current];
std::optional<std::size_t> best;
double best_score = -std::numeric_limits<double>::infinity();
for (const std::size_t child_index : parent.children) {
const Node &child = nodes_[child_index];
if (child.closed) continue;
const double score = selection_score(parent, child);
if (!best.has_value() || score > best_score ||
(score == best_score && child.action_key < nodes_[*best].action_key)) {
best = child_index;
best_score = score;
}
}
if (!best.has_value()) return std::nullopt;
current = *best;
path.push_back(current);
}
return path;
}

void backup(const std::vector<std::size_t> &path) {
for (auto iterator = path.rbegin(); iterator != path.rend(); ++iterator) {
refresh(*iterator);
}
}

SearchResult make_result() const {
if (nodes_.empty() || nodes_[0].children.empty()) {
impossible("root action");
}
const Node &root = nodes_[0];
std::size_t best = root.children.front();
double best_score = -std::numeric_limits<double>::infinity();
SearchResult result;
result.root_actions.reserve(root.children.size());
for (const std::size_t child_index : root.children) {
const Node &child = nodes_[child_index];
const float action_value = -child.value;
const double score = final_action_score(action_value, child.visits);
if (score > best_score ||
(score == best_score && child.action_key < nodes_[best].action_key)) {
best = child_index;
best_score = score;
}
result.root_actions.push_back(RootActionStat{
child.action_key, action_value, -child.prior_value, child.visits,
child.selection_visits, child.tactical});
}
const Node &chosen = nodes_[best];
result.moves = chosen.incoming_action;
result.encoded = chosen.action_key;
result.value = -chosen.value;
result.solved = root.solved;
if (root.solved) {
result.solved_winner =
root.value > 0.0F ? root.perspective : opponent(root.perspective);
}
result.stats = stats_;
return result;
}
};

SearchResult choose_complete_turn(const GameState &state, SearchConfig config) {
BestFirstMinimaxSearch search(state, config);
return search.run();
}

void apply_encoded_turn(GameState &state, std::string_view encoded) {
if (encoded.empty()) invalid("empty complete turn");
GameState next = state;
const Player mover = next.to_move;
for (const char direction : encoded) {
if (is_terminal(next) || next.to_move != mover) {
invalid("action overrun");
}
next = apply_move(next, decode_direction(next.ball, direction));
}
if (!is_terminal(next) && next.to_move == mover) {
invalid("incomplete action");
}
state = std::move(next);
}

std::string choose_complete_turn(GameState &state,
std::uint32_t search_time_ms) {
SearchConfig config;
config.absolute_deadline =
SearchClock::now() + std::chrono::milliseconds(search_time_ms);
SearchResult result = choose_complete_turn(
static_cast<const GameState &>(state), config);
apply_encoded_turn(state, result.encoded);
return result.encoded;
}

}  // namespace papersoccer::jacek_native_bfm

#if !defined(PAPER_SOCCER_JACEK_NATIVE_BFM_NO_MAIN) && !defined(JACEK_NATIVE_NO_MAIN)
int main() {
using namespace papersoccer;
using namespace papersoccer::jacek_native_bfm;
std::ios::sync_with_stdio(false);
std::cin.tie(nullptr);

int player_id = -1;
if (!(std::cin >> player_id)) return 0;
Player me;
try {
me = player_for_id(player_id);
} catch (const std::exception &) {
return 1;
}
std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

RulesConfig rules;
rules.goal_rule = kProductionGoalRule;
rules.blocked_rule = kProductionBlockedRule;
GameState state = make_initial_state(rules);
bool first_execution = true;
while (true) {
int opponent_length = 0;
if (!(std::cin >> opponent_length)) break;
std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
std::string opponent_move;
if (!std::getline(std::cin, opponent_move)) break;
if (opponent_move != "-" &&
opponent_length != static_cast<int>(opponent_move.size())) {
return 1;
}
try {
if (opponent_move == "-") {
if (player_id != 0 || !first_execution) return 1;
} else {
apply_encoded_turn(state, opponent_move);
}
if (is_terminal(state)) return 0;
if (state.to_move != me) return 1;
const std::uint32_t budget =
first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs;
const std::string action = choose_complete_turn(state, budget);
std::cout << action << std::endl;
first_execution = false;
} catch (const std::exception &) {
return 1;
}
}
return 0;
}
#endif
