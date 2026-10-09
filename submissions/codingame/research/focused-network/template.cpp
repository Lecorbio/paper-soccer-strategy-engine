#include <ctime>
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC optimize("O3")
#endif
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <optional>
#include <array>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <string_view>
#include <span>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <limits>
#include <string>
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
}   
namespace papersoccer {
bool is_regular_point(const RulesConfig &config, Point point);
bool is_goal_point(const RulesConfig &config, Point point);
bool is_attacking_goal(const RulesConfig &config, Point point, Player player);
bool is_boundary_point(const RulesConfig &config, Point point);
bool is_neighbor(Point from, Point to);
bool is_forbidden_boundary_segment(const RulesConfig &config, Segment segment);
std::vector<Point> neighbors(const RulesConfig &config, Point from, Player player);
}   
namespace papersoccer {
GameState make_initial_state(const RulesConfig &config = {});
std::vector<Move> legal_moves(const GameState &state);
bool grants_extra_turn(const GameState &before, Point destination);
GameState apply_move(const GameState &state, Move move);
bool is_terminal(const GameState &state);
std::optional<Player> winner(const GameState &state);
}   
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
}   
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
}   
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
}   
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
}   
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
}   
namespace papersoccer::turn_action_v2::learned_codec {
inline std::vector<std::uint8_t> decode(
std::span<const std::uint8_t> input, std::size_t count, unsigned width,
const std::array<std::uint8_t, 16>& lengths) {
if ((width != 3U && width != 4U) || count > input.size() * 8U ||
count > (SIZE_MAX - 7U) / width) {
throw std::invalid_argument("representation dimensions");
}
std::array<unsigned, 16> counts{}, first{}, offsets{}, symbols{};
unsigned total = 0;
for (unsigned symbol = 0; symbol < lengths.size(); ++symbol) {
const unsigned length = lengths[symbol];
if (length > 15U || (length && (symbol >= (1U << width) || symbol == (1U << (width - 1U))))) {
throw std::invalid_argument("representation code lengths");
}
if (length) { ++counts[length]; ++total; }
}
if (total == 0U) throw std::invalid_argument("representation empty alphabet");
unsigned code = 0, offset = 0;
for (unsigned length = 1; length <= 15; ++length) {
code = (code + counts[length - 1U]) << 1U;
first[length] = code;
offsets[length] = offset;
if (code + counts[length] > (1U << length)) {
throw std::invalid_argument("representation oversubscribed codes");
}
for (unsigned symbol = 0; symbol < lengths.size(); ++symbol) {
if (lengths[symbol] == length) symbols[offset++] = symbol;
}
}
std::vector<std::uint8_t> output((count * width + 7U) / 8U, 0);
std::size_t cursor = 0;
for (std::size_t index = 0; index < count; ++index) {
unsigned current = 0, symbol = 16;
for (unsigned length = 1; length <= 15; ++length) {
if (cursor / 8U >= input.size()) throw std::invalid_argument("representation truncated stream");
current = (current << 1U) | ((input[cursor / 8U] >> (cursor % 8U)) & 1U);
++cursor;
if (current >= first[length] && current - first[length] < counts[length]) {
symbol = symbols[offsets[length] + current - first[length]];
break;
}
}
if (symbol == 16) throw std::invalid_argument("representation invalid prefix");
const std::size_t bit = index * width;
output[bit / 8U] |= static_cast<std::uint8_t>(symbol << (bit % 8U));
if (bit % 8U + width > 8U) {
output[bit / 8U + 1U] |= static_cast<std::uint8_t>(symbol >> (8U - bit % 8U));
}
}
if ((cursor + 7U) / 8U != input.size() ||
(cursor % 8U && (input.back() >> (cursor % 8U)))) {
throw std::invalid_argument("representation trailing data");
}
return output;
}
}   
namespace papersoccer::turn_action_v2::learned_model {
inline constexpr std::size_t kInputs = 6301;
inline constexpr std::size_t kHiddenOne = 12;
inline constexpr std::size_t kHiddenTwo = 8;
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = 75716;
inline constexpr std::size_t kPackedByteCount = 37858;
inline constexpr float kScaleOne = 0.0400825329F;
inline constexpr float kScaleTwo = 0.0803351477F;
inline constexpr float kScaleThree = 0.0364000015F;
inline constexpr unsigned kWeightBits = 4;
inline constexpr std::array<std::uint8_t, 16> kHuffmanLengths{1,8,3,10,5,11,8,11,0,10,6,10,4,9,2,9};
inline constexpr bool kBootstrapZero = false;
inline constexpr std::string_view kRuntimeSchema = "papersoccer.compact-representation.runtime.v1";
inline constexpr std::string_view kFeatureSchema = "papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:mover-relative-rotate180:true-turn-distance+free-degree";
inline constexpr std::string_view kPayloadSha256 = "759619d0644c1bbbb9abce2e9e2c89e5e0db36600a824896012d471ca110f2d1";
inline constexpr std::string_view kRuntimeBodySha256 = "82313264c0f506c41257d64fc0312eaa8aefb484cb225e4383b0afd71d250a33";
inline constexpr std::string_view kIdentity = "repr-h12-b4-82313264c0f5";
inline constexpr std::string_view kPackedWeights =
"A4Ddz7f+8b0mn/QNfm6n78fe/bTW9/N38q1/fD+xfy66v//8qFjf3/vHV7tf374/96fmb9/Pn5vVKEDIu/uv+fHtms/rr/6x"
"u5q/3d9b99368e0PAAAAAKR3LfnZrtavT7pa618BQLyL7udxf7/X375/fqZ6Pd87V7f72/eTLye+UX8/5lsA4Pd8+se3s3ys"
"b3W3j/nG+B6+kZ9bf92PH1du+96fs59b7Wb8vR9/+35y+vjrg3+67W8f+9b3su/nh1388fknabh3y9/7+Xn3/fO35p52/fPX"
"+4vk7X1ure9dpifX+Pn7+fnpG/65qDN0/PH5x/7pXrdv/8z+xt/uZ286/fT9/PSlsQ/nb+vv7h/6ytntfNzDx5wT3667fejb"
"X5fT61vm54ajndvx7fDPZbu/MrdrH2fv69X9N7ufb63v3Xn79Lk/w88Z55/Ou16VrFY7zE7tmuxaroefe/EtTru5d56fHPfX"
"3vf33vf2/X7Tz73a95OXXM6874f7637PrB5n8urd6We10+3M1GoHZMlze1/LtfDpc3/+od2aT47b2Tfazzf2mXGZiz637+17"
"Z71vxzfWPuZj8sH3bvv27d6+XaPd9u3sfMu+98U/H1zYPtYn0tpu19o32yftffDRkSyb05pvdb/7/Fbfs9x6+97Znff9+na9"
"Zs23s3Pr7VP3u+u37vs9VJa1tOxW2c180L41n9r2oWuZ29m3tB266Hzr7Vvv+/bt+923q9u35pu+9jHfzto388lu347v9bHP"
"ugt1u+2b2/bBx2zf/Pput+S845MPvtVOu/lW2cftVuwj1zbv69Vneft6fO2+X9dvutfWt7VvO7/7+u3ue93v73e/dd/v9fb1"
"y32/33739Vt9v/CZfda2r8c1a+ennZz3mT7LFjlTuOdz2Ddv3+Q0cwvlvFMsbodu35rvRdltp7t2VnfDvtm+tbOvp++enXvZ"
"t0jYiXbko/FZLrKvdy6yz/v6GhwcW93a+xb5OCfS7vl21/5+2e5sn+99b9/vBx/6nr27+/3m+/2yalTJN+tj++p9b9/7LO2e"
"c8bBYe2jHVyt3/etzj7vc727NDnSrndqlTOyldiu2D74trnutX0L7+vpu2m+nnc35SwX+Mb7LNvn3e428z3Pd4PDvmbb5+X7"
"tvGZ95nayQ7ah/dNeZ+1L7Rix3a1b/t+P3mfd669V/fud9/vp/nu5ettV+97+943173v+X6vvM/Ld8ut3E4/a8nS8T7brtnt"
"xr4Zvu3ts22+2T6z7evZd5jDvt5sZ7cOc712Bt/cvkbO0rG/vr3eF/JhX7rt692+e2xfj+8e9jWu17t92JfwGV9rnxM3W/u2"
"7WvTPrzPtt7n6dv2zu22b9l8XvuM29fy5Wxfr93tevv6/fZ9j/m89rVr+3q3+0b7WjvbPrfbbk37vO1rT+8zvt7WPvY+5iT7"
"sL1rc02c5bvZOexOvX2e3a3Zeb1L2df73D34lO8s+8Y+Q5+dL20+csb2bfk2+HZ+v47t89rXtH29231rfMb1uGvttXN7n5fP"
"M33MoXzoe9a9z+Mz9b49fVvp8+prfC9Na9rn8WHuvPraeJ/Hx3Dik1U773azPptvlvd554vW59lZq5b3+TRfa/dtt/d5+W7M"
"t+Vr8M18VnPka1nXyxF9lmvFZ+drEGDNZ+ezxbfxHebb+Aw+L1+b7fPytdk+6oP1WX0nObVDz9fvdfdczpLm+Tzf2TlkOux7"
"rz5vdzs+d+pMe7r1+sx9N57P73W9eTurj41iJ/i868vM57XvtuPb8pnss/q8mW+7vibOPtdsfcY1AKzm7Hy2+KyuAgDrWhcA"
"AAAAAAL2+qc1LQAAAwAABAAAAADAzr75e58fAAACACQgAAAAYABMtx++AAAAAMAAAAAAAADMfz/jjwEAAACAAAAAAAAGxM3c"
"DwACAAYAAQAQAMCAwWqjBgAAgACAwQAAA0AAs7s3KwAAAGACAAAAAAQAIfRtzd/5+3f+CQD+x453BQAMAAAAAABo/Rzr2w8A"
"4J/PP+4MAAAAAAAQAGx8wweAkP5O/gAGAoIAAAAAYOODbxCA1m1zfwEAAAAAAADA+FrcBgC2neXbAAZEAAAAgAHwnM6zAMis"
"SAEAGAAAAQBgYM7y2wCAXr1HBjCAAGYAgAEA6jMfvfP8vdnrBzTgn/rnb/v5/u4vYAAADAAAADbIT/O3+/sPQBg+3AcAAAAA"
"EAAAgGP7FgBEZwkAQAAAgAEAgJ1v7CYAJn3DmQEIYAAAAgBAo3QhAGDLFAAIMAAAoAAEOP1+a09AIK8z6wYAAAAAAIABaLGd"
"AUCPQhUgAAAAgAEAIOESACAAfu73///++f+/37///l8A+t+/////v//7/7///vvv/wEAAAYAAACAHze8L5cxYONj3xkAAYYB"
"BgAAALq7xm4fAOC+bW4AAAAAAIYAAPQ12+1jAIab8xkAAAAAAAAAeD7qnQBg2Q0HwAAAAAAAAGD75vYIALLCCgAAAAACAADY"
"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";
}   
namespace papersoccer::turn_action_v2::learned_eval {
struct Features {
std::array<std::uint16_t, 421> indices{};
std::uint16_t count{};
};
inline int base64_value(char c) noexcept {
if (c >= 'A' && c <= 'Z') return c - 'A';
if (c >= 'a' && c <= 'z') return c - 'a' + 26;
if (c >= '0' && c <= '9') return c - '0' + 52;
if (c == '+') return 62;
if (c == '/') return 63;
return -1;
}
inline std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
if (encoded.empty() || encoded.size() % 4U) throw std::invalid_argument("learned model base64 length");
std::vector<std::uint8_t> bytes;
bytes.reserve(encoded.size() / 4U * 3U);
for (std::size_t offset = 0; offset < encoded.size(); offset += 4U) {
const bool last = offset + 4U == encoded.size();
const char c0 = encoded[offset], c1 = encoded[offset + 1U];
const char c2 = encoded[offset + 2U], c3 = encoded[offset + 3U];
const int a = base64_value(c0), b = base64_value(c1);
const int c = c2 == '=' ? 0 : base64_value(c2);
const int d = c3 == '=' ? 0 : base64_value(c3);
if (a < 0 || b < 0 || c < 0 || d < 0 || c0 == '=' || c1 == '=' ||
(!last && (c2 == '=' || c3 == '=')) || (c2 == '=' && c3 != '='))
throw std::invalid_argument("learned model base64 encoding");
bytes.push_back(static_cast<std::uint8_t>((a << 2U) | (b >> 4U)));
if (c2 == '=') {
if (b & 15) throw std::invalid_argument("learned model base64 padding");
continue;
}
bytes.push_back(static_cast<std::uint8_t>((b << 4U) | (c >> 2U)));
if (c3 == '=') {
if (c & 3) throw std::invalid_argument("learned model base64 padding");
continue;
}
bytes.push_back(static_cast<std::uint8_t>((c << 6U) | d));
}
return bytes;
}
inline const std::vector<std::int8_t> &weights() {
static const std::vector<std::int8_t> result = [] {
static_assert(learned_model::kInputs == 6301 && learned_model::kHiddenOne == 12 &&
learned_model::kHiddenTwo == 8 && learned_model::kWeightBits == 4);
const auto compressed = decode_base64(learned_model::kPackedWeights);
const auto packed = learned_codec::decode(compressed, learned_model::kWeightCount,
learned_model::kWeightBits, learned_model::kHuffmanLengths);
if (packed.size() != learned_model::kPackedByteCount)
throw std::invalid_argument("learned model packed size");
std::vector<std::int8_t> decoded(learned_model::kWeightCount);
for (std::size_t index = 0; index < decoded.size(); ++index) {
const int code = (packed[index / 2U] >> ((index % 2U) * 4U)) & 15;
if (code == 8) throw std::invalid_argument("learned model reserved code");
decoded[index] = static_cast<std::int8_t>(code >= 8 ? code - 16 : code);
}
return decoded;
}();
return result;
}
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
inline float fast_tanh(float value) noexcept {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
if (value < -4.95F) return -1.0F;
if (value > 4.95F) return 1.0F;
const float square = value * value;
return value * (135135.0F + square * (17325.0F + square * (378.0F + square))) /
(135135.0F + square * (62370.0F + square * (3150.0F + 28.0F * square)));
}
inline float evaluate(const Features &features) {
const auto &model = weights();
std::array<std::int32_t, 12> first{};
for (std::size_t index = 0; index < features.count; ++index) {
const std::size_t offset = features.indices[index] * 12U;
for (std::size_t hidden = 0; hidden < 12; ++hidden)
first[hidden] += static_cast<int>(model[offset + hidden]);
}
std::array<float, 12> activated{};
for (std::size_t index = 0; index < 12; ++index) {
const float value = static_cast<float>(first[index]) * learned_model::kScaleOne;
activated[index] = value < 0.0F ? 0.01F * value : value * value;
}
std::array<float, 8> second{};
constexpr std::size_t offset_two = 6301U * 12U;
for (std::size_t input = 0; input < 12; ++input) {
for (std::size_t hidden = 0; hidden < 8; ++hidden) {
volatile float scaled = activated[input] * learned_model::kScaleTwo;
volatile float term = scaled * static_cast<float>(model[offset_two + input * 8U + hidden]);
second[hidden] = second[hidden] + term;
}
}
for (auto &value : second) value = value < 0.0F ? 0.01F * value : value;
constexpr std::size_t offset_three = offset_two + 12U * 8U;
float output = 0.0F;
for (std::size_t hidden = 0; hidden < 8; ++hidden) {
volatile float scaled = second[hidden] * learned_model::kScaleThree;
volatile float term = scaled * static_cast<float>(model[offset_three + hidden]);
output = output + term;
}
return fast_tanh(output);
}
}   
namespace papersoccer::turn_action_v2 {
constexpr std::uint32_t kFirstSearchTimeMs = 990;
constexpr std::uint32_t kLaterSearchTimeMs = 180;
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
enum class LeafMix { PureLearned, HalfHand };
constexpr LeafMix kDefaultLeafMix = LeafMix::PureLearned;
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
LeafMix leaf_mix{kDefaultLeafMix};
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
int hand_score{};
float learned_value{};
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
queue_(topology_->vertex_count()) {
if (config_.max_turn_depth == 0 ||
config_.max_turn_depth > kMaximumTurnDepth || config_.max_nodes == 0) {
throw std::invalid_argument("invalid complete-turn search configuration");
}
if (state.config.width != 8 || state.config.height != 10 ||
state.config.goal_rule != GoalRule::OwnGoalsAllowed ||
state.config.blocked_rule != BlockedRule::MoverLoses)
throw std::invalid_argument("learned evaluator requires contest rules");
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
private:
SearchConfig config_;
std::shared_ptr<const detail::SearchTopology> topology_;
detail::SearchPosition position_;
TranspositionTable table_;
EvaluationTable evaluations_;
SearchStats stats_;
std::vector<int> distances_;
std::vector<detail::SearchTopology::VertexIndex> queue_;
std::deque<detail::SearchTopology::VertexIndex> distance_queue_;
std::array<std::uint8_t,128> turn_queue_{};
std::vector<detail::SearchTopology::VertexIndex> rotated_vertices_;
std::vector<detail::SearchTopology::EdgeIndex> rotated_edges_;
std::optional<SearchClock::time_point> deadline_;
std::vector<Move> current_action_;
std::vector<Move> captured_action_;
int captured_score_{};
bool captured_any_{};
void visit_node() {
if (stats_.nodes >= config_.max_nodes) {
throw SearchBudgetReached{};
}
if (deadline_.has_value() &&
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
struct TurnDistanceGraph {
struct Arc {std::uint16_t edge;std::uint8_t destination;};
std::array<std::array<Arc,8>,105> arcs{};
std::array<std::uint8_t,105> count{},fixed_zero{};
explicit TurnDistanceGraph(const detail::SearchTopology &t) {
if(t.vertex_count()!=105||t.edge_count()!=316)throw std::logic_error("turn topology");
for(std::size_t v=0;v<105;++v) {
const auto &a=t.adjacency(v,Player::One);count[v]=a.count;
const Point p=t.point(v);fixed_zero[v]=is_boundary_point(t.config(),p)||is_goal_point(t.config(),p);
for(std::uint8_t i=0;i<a.count;++i)arcs[v][i]={static_cast<std::uint16_t>(a.arcs[i].edge),static_cast<std::uint8_t>(a.arcs[i].destination)};
}
}
};
void calculate_goal_turn_distances(Player player) {
static const TurnDistanceGraph graph(*topology_);
std::array<std::uint8_t,105> cost;
for(std::size_t v=0;v<105;++v)cost[v]=!(graph.fixed_zero[v]||position_.vertex_visited(v));
std::fill(distances_.begin(),distances_.end(),1'000'000);
const auto start=position_.ball_vertex();distances_[start]=0;
unsigned head=0,count=1;turn_queue_[0]=static_cast<std::uint8_t>(start);
while(count) {
const auto vertex=turn_queue_[head];head=(head+1U)&127U;--count;
for(std::uint8_t i=0;i<graph.count[vertex];++i) {
const auto arc=graph.arcs[vertex][i];
if(position_.edge_used(arc.edge))continue;
const int step=cost[arc.destination],distance=distances_[vertex]+step;
if(distance>=distances_[arc.destination])continue;
distances_[arc.destination]=distance;
if(count==turn_queue_.size()){calculate_goal_turn_distances_slow(player);return;}
if(step==0){head=(head+127U)&127U;turn_queue_[head]=arc.destination;}
else turn_queue_[(head+count)&127U]=arc.destination;
++count;
}
}
}
void calculate_goal_turn_distances_slow(Player player) {
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
int legacy_hand_score() {
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
return hand_score;
}
learned_eval::Features learned_features() {
if (topology_->edge_count() != 316 || topology_->vertex_count() != 105)
throw std::logic_error("learned evaluator topology mismatch");
learned_eval::Features features;
const Player mover = position_.to_move();
const bool rotate = mover == Player::Two;
for (std::size_t canonical = 0; canonical < 316; ++canonical) {
const auto physical = rotate ? rotated_edges_[canonical] : canonical;
if (position_.edge_used(static_cast<detail::SearchTopology::EdgeIndex>(physical)))
features.indices[features.count++] = static_cast<std::uint16_t>(canonical);
}
calculate_goal_turn_distances(mover);
for (std::size_t canonical = 0; canonical < 105; ++canonical) {
const auto physical = rotate ? rotated_vertices_[canonical] : canonical;
int category = 56;
if (distances_[physical] < 7) {
int free_degree = 0;
const auto &adjacency = topology_->adjacency(
static_cast<detail::SearchTopology::VertexIndex>(physical), mover);
for (std::uint8_t slot = 0; slot < adjacency.count; ++slot)
free_degree += !position_.edge_used(adjacency.arcs[slot].edge);
category = distances_[physical] * 8 + std::clamp(free_degree - 1, 0, 7);
}
features.indices[features.count++] = static_cast<std::uint16_t>(316 + canonical * 57 + category);
}
return features;
}
EvaluationSnapshot make_evaluation_snapshot() {
EvaluationSnapshot snapshot;
snapshot.mover_sign = player_sign(position_.to_move());
snapshot.learned_value = learned_eval::evaluate(learned_features());
const int learned_scale = config_.leaf_mix == LeafMix::HalfHand ? 1500 : kMaximumEvaluation;
const int learned_score = snapshot.mover_sign * static_cast<int>(
std::lround(snapshot.learned_value * static_cast<float>(learned_scale)));
int score = learned_score;
if (config_.leaf_mix == LeafMix::HalfHand) {
snapshot.hand_score = legacy_hand_score();
score = (snapshot.hand_score + learned_score) / 2;
}
snapshot.anchor_score = std::clamp(score, -kMaximumEvaluation, kMaximumEvaluation);
return snapshot;
}
int evaluate() {
++stats_.leaf_evaluations;
return make_evaluation_snapshot().anchor_score;
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
return left.move.to.x < right.move.to.x;
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
return cached_evaluate();
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
SearchClock::time_point response_deadline) {
SearchConfig config;
config.absolute_deadline = response_deadline;
config.max_nodes = std::numeric_limits<std::uint64_t>::max();
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
std::string choose_complete_turn(GameState &state,
std::uint32_t search_time_ms) {
return choose_complete_turn(state,
SearchClock::now() + std::chrono::milliseconds(search_time_ms));
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
std::cin >> std::ws;
if (std::cin.peek() == std::char_traits<char>::eof()) {
break;
}
const auto response_started = SearchClock::now();
auto response_deadline = response_started + std::chrono::milliseconds(
first_execution ? kFirstSearchTimeMs : kLaterSearchTimeMs);
if (first_execution) {
const std::clock_t startup_cpu = std::clock();
if (startup_cpu == static_cast<std::clock_t>(-1) || startup_cpu < 0) {
response_deadline = response_started;
} else {
const double startup_ms = 1000.0 * static_cast<double>(startup_cpu) /
static_cast<double>(CLOCKS_PER_SEC);
response_deadline -= std::chrono::duration_cast<SearchClock::duration>(
std::chrono::duration<double, std::milli>(startup_ms));
}
}
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
std::string action;
if (!try_replay_correction(state, player_id, transcript, action)) {
action = choose_complete_turn(state, response_deadline);
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
