
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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace compact_value_bfm::model {

inline constexpr std::size_t kInputs = 6301;
inline constexpr std::size_t kHiddenOne = 8;
inline constexpr std::size_t kHiddenTwo = 8;
inline constexpr std::size_t kOutputs = 1;
inline constexpr std::size_t kWeightCount = 50'480;
inline constexpr std::size_t kPackedByteCount = 0;
inline constexpr float kScaleOne = 1.0F;
inline constexpr float kScaleTwo = 1.0F;
inline constexpr float kScaleThree = 1.0F;
inline constexpr bool kBootstrapZero = true;
inline constexpr std::string_view kRuntimeSchema =
"papersoccer.compact-value-bfm-runtime.v1";
inline constexpr std::string_view kFeatureSchema =
"papersoccer.jacek-replay-bfm.features.v1:edge316+vertex105x57:"
"mover-relative-rotate180:true-turn-distance+free-degree";
inline constexpr std::string_view kPayloadSha256 =
"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
inline constexpr std::string_view kRuntimeBodySha256 =
"0000000000000000000000000000000000000000000000000000000000000000";
inline constexpr std::string_view kIdentity = "bootstrap-zero-not-qualified";
inline constexpr std::string_view kPackedWeights = "";

}  // namespace compact_value_bfm::model

namespace compact_value_bfm {

inline constexpr int kVertexCount = 105;
inline constexpr int kEdgeCount = 316;
inline constexpr int kVertexCategories = 57;
inline constexpr int kFeatureCount =
kEdgeCount + kVertexCount * kVertexCategories;
inline constexpr int kMaximumActiveFeatures = kEdgeCount + kVertexCount;
inline constexpr int kMaximumActionLength = kEdgeCount;
inline constexpr std::size_t kMaximumActions = 250;
inline constexpr std::size_t kMaximumPartialPaths = 50'000;
inline constexpr std::size_t kRootPartialPaths = 4'000;
inline constexpr std::size_t kNonrootPartialPaths = 512;
inline constexpr std::size_t kMaximumTreeNodes = 120'000;
inline constexpr std::size_t kProductionTreeNodes = 80'000;
inline constexpr std::uint64_t kMaximumExpansions = 2'000'000;
inline constexpr std::size_t kProofPaths = 64;
inline constexpr double kExploration = 0.95;
inline constexpr double kFirstPlayUrgency = 0.5;
inline constexpr double kFinalVisitWeight = 1.0;
inline constexpr float kMateScore = 10'000.0F;

static_assert(kFeatureCount == 6301);

struct Action {
std::array<std::uint8_t, kMaximumActionLength> directions{};
std::uint16_t length{};

bool operator==(const Action &other) const noexcept;
std::string text() const;
};

struct State {
std::array<std::uint64_t, 5> used{};
std::uint8_t ball{};
std::uint8_t to_move{};
std::int8_t winner{-1};
std::uint16_t ply{};

bool terminal() const noexcept { return winner >= 0; }
};

struct SparseFeatures {
std::array<std::uint16_t, kMaximumActiveFeatures> indices{};
std::uint16_t count{};

bool operator==(const SparseFeatures &) const noexcept = default;
};

class Topology {
public:
struct Arc {
std::uint16_t edge{};
std::uint8_t destination{};
std::uint8_t direction{};
};

static const Topology &get();
int x(int vertex) const noexcept { return xs_[vertex]; }
int y(int vertex) const noexcept { return ys_[vertex]; }
int vertex_at(int x, int y) const noexcept;
int degree(int vertex) const noexcept { return degrees_[vertex]; }
Arc arc(int vertex, int index) const noexcept { return arcs_[vertex][index]; }
int rotated_vertex(int vertex) const noexcept {
return rotated_vertices_[vertex];
}
int rotated_edge(int edge) const noexcept { return rotated_edges_[edge]; }
bool boundary(int vertex) const noexcept { return boundaries_[vertex]; }
bool north_goal(int vertex) const noexcept { return ys_[vertex] == 0; }
bool south_goal(int vertex) const noexcept { return ys_[vertex] == 12; }

private:
Topology();
std::array<std::int8_t, kVertexCount> xs_{};
std::array<std::int8_t, kVertexCount> ys_{};
std::array<std::array<Arc, 8>, kVertexCount> arcs_{};
std::array<std::uint8_t, kVertexCount> degrees_{};
std::array<std::uint8_t, kVertexCount> rotated_vertices_{};
std::array<std::uint16_t, kEdgeCount> rotated_edges_{};
std::array<bool, kVertexCount> boundaries_{};
};

State initial_state();
bool same_boundary(const State &left, const State &right) noexcept;
bool edge_used(const State &state, int edge) noexcept;
bool vertex_visited(const State &state, int vertex) noexcept;
std::uint8_t legal_arcs(const State &state,
std::array<Topology::Arc, 8> &result) noexcept;
bool apply_edge(State &state, std::uint8_t direction) noexcept;
bool apply_action(State &state, const Action &action,
bool require_complete = true) noexcept;
State rotate_and_swap(const State &state);
std::uint64_t state_hash(const State &state) noexcept;
std::uint64_t canonical_state_hash(const State &state) noexcept;
SparseFeatures active_features(const State &state);
SparseFeatures active_features(const State &state, std::uint8_t perspective);

std::string sha256_hex(std::span<const std::uint8_t> bytes);
float first_activation(float value) noexcept;
float second_activation(float value) noexcept;
float fast_tanh(float value) noexcept;

struct ModelDescriptor {
std::size_t inputs{kFeatureCount};
std::size_t hidden_one{};
std::size_t hidden_two{};
float scale_one{};
float scale_two{};
float scale_three{};
std::string_view packed_base64{};
std::string_view packed_sha256{};
bool allow_empty_bootstrap{};
};

struct PreparedEvaluation {
SparseFeatures features{};
std::array<std::int32_t, 12> first{};
};

class QuantizedModel {
public:
explicit QuantizedModel(ModelDescriptor descriptor);

PreparedEvaluation prepare(const SparseFeatures &features) const noexcept;
float evaluate(const SparseFeatures &features) const noexcept;
float evaluate_delta(const PreparedEvaluation &base,
const SparseFeatures &features) const noexcept;
float finish(std::array<std::int32_t, 12> first) const noexcept;
std::span<const std::int8_t> weights() const noexcept { return weights_; }
std::size_t hidden_one() const noexcept { return hidden_one_; }
std::size_t hidden_two() const noexcept { return hidden_two_; }
std::string_view payload_sha256() const noexcept { return payload_sha256_; }

private:
std::vector<std::int8_t> weights_{};
std::size_t hidden_one_{};
std::size_t hidden_two_{};
float scale_one_{};
float scale_two_{};
float scale_three_{};
std::string payload_sha256_{};

void add_input(std::array<std::int32_t, 12> &first, std::size_t input,
int multiplier) const noexcept;
};

const QuantizedModel &deployment_model();

enum class TacticalClass : std::uint8_t {
ImmediateGoal,
ForcedCutoff,
SafeHandoff,
OpponentImmediateGoal,
OwnGoal,
};

struct GeneratorConfig {
std::size_t max_actions{kMaximumActions};
std::size_t max_partial_paths{kRootPartialPaths};
std::chrono::steady_clock::time_point deadline{
std::chrono::steady_clock::time_point::max()};
std::uint64_t shuffle_seed{0x6a09e667f3bcc909ULL};
const Action *precomputed_emergency{};
};

struct GeneratorStats {
std::uint64_t partial_paths{};
std::uint64_t proof_paths{};
std::uint64_t completed_actions{};
std::uint64_t duplicate_boundaries{};
std::uint64_t fifo_extractions{};
std::uint64_t lifo_extractions{};
std::uint64_t tactical_actions{};
std::uint64_t truncations{};
std::uint64_t cap_drops{};
bool deadline_reached{};
bool proof_truncated{};
};

struct CompleteTurnAction {
State result{};
Action action{};
TacticalClass tactical{TacticalClass::SafeHandoff};
std::uint32_t order{};
};

struct GenerationResult {
std::vector<CompleteTurnAction> actions{};
GeneratorStats stats{};
bool exhaustive{};
};

Action emergency_complete_action(
const State &state,
std::uint64_t seed = GeneratorConfig{}.shuffle_seed);
GenerationResult generate_complete_turns(const State &state,
const GeneratorConfig &config = {});

struct SearchConfig {
std::size_t max_actions{kMaximumActions};
std::size_t root_partial_paths{kRootPartialPaths};
std::size_t nonroot_partial_paths{kNonrootPartialPaths};
std::size_t max_tree_nodes{kProductionTreeNodes};
std::uint64_t max_expansions{kMaximumExpansions};
std::uint64_t shuffle_seed{GeneratorConfig{}.shuffle_seed};
double exploration{kExploration};
double fpu{kFirstPlayUrgency};
double final_visit_weight{kFinalVisitWeight};
const QuantizedModel *model{};
};

struct SearchStats {
std::uint64_t expansions{};
std::uint64_t generated_children{};
std::uint64_t evaluated_children{};
std::uint64_t tactical_children{};
std::uint64_t generator_partial_paths{};
std::uint64_t generator_proof_paths{};
std::uint64_t duplicate_boundaries{};
std::uint64_t fifo_extractions{};
std::uint64_t lifo_extractions{};
std::size_t tree_nodes{};
std::uint32_t max_depth{};
bool deadline_reached{};
bool tree_cap_reached{};
bool expansion_cap_reached{};
};

struct RootActionStat {
Action action{};
TacticalClass tactical{TacticalClass::SafeHandoff};
float value{};
float initial_value{};
std::uint32_t visits{};
std::uint32_t selection_visits{};
bool solved{};
std::uint32_t order{};
};

struct SearchResult {
Action action{};
float value{};
bool solved{};
std::vector<RootActionStat> root_actions{};
SearchStats stats{};
};

float mate_value(int winner, int perspective, std::uint32_t depth) noexcept;
double uct_score(float value, std::uint32_t parent_visits,
std::uint32_t child_visits, double exploration,
double fpu);
double final_score(float value, std::uint32_t visits, double weight);
SearchResult search(
const State &state, std::chrono::steady_clock::time_point deadline,
const SearchConfig &config = {}, const Action *precomputed_emergency = nullptr);

}  // namespace compact_value_bfm

namespace compact_value_bfm {
namespace {

static_assert(model::kInputs == kFeatureCount && model::kOutputs == 1);

constexpr std::array<int, 8> kDx{0, 1, 1, 1, 0, -1, -1, -1};
constexpr std::array<int, 8> kDy{-1, -1, 0, 1, 1, 1, 0, -1};

bool regular(int x, int y) noexcept {
return x >= 0 && x <= 8 && y >= 1 && y <= 11;
}

bool goal(int x, int y) noexcept {
return x >= 3 && x <= 5 && (y == 0 || y == 12);
}

bool boundary_coordinate(int x, int y) noexcept {
return regular(x, y) &&
(x == 0 || x == 8 || ((y == 1 || y == 11) && x != 4));
}

bool permitted_edge(int ax, int ay, int bx, int by) noexcept {
const int dx = std::abs(ax - bx);
const int dy = std::abs(ay - by);
if (std::max(dx, dy) != 1) return false;
if (regular(ax, ay) && regular(bx, by)) {
if (boundary_coordinate(ax, ay) && boundary_coordinate(bx, by)) {
if (ay == by && (ay == 1 || ay == 11)) return false;
if (ax == bx && (ax == 0 || ax == 8)) return false;
}
return true;
}
if (goal(ax, ay) == goal(bx, by)) return false;
const int field_x = regular(ax, ay) ? ax : bx;
const int field_y = regular(ax, ay) ? ay : by;
const int goal_x = goal(ax, ay) ? ax : bx;
const int goal_y = goal(ax, ay) ? ay : by;
return field_x >= 3 && field_x <= 5 &&
((field_y == 1 && goal_y == 0) ||
(field_y == 11 && goal_y == 12)) &&
!(field_x == goal_x && (field_x == 3 || field_x == 5));
}

int direction_for(int dx, int dy) noexcept {
for (int direction = 0; direction < 8; ++direction) {
if (kDx[direction] == dx && kDy[direction] == dy) return direction;
}
return -1;
}

std::uint64_t mix64(std::uint64_t value) noexcept {
value += 0x9e3779b97f4a7c15ULL;
value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
return value ^ (value >> 31U);
}

bool append(Action &action, std::uint8_t direction) noexcept {
if (action.length >= action.directions.size()) return false;
action.directions[action.length++] = direction;
return true;
}

void set_edge(State &state, int edge) noexcept {
state.used[edge >> 6] |= 1ULL << (edge & 63);
}

bool valid_sha256(std::string_view value) noexcept {
return value.size() == 64 &&
std::all_of(value.begin(), value.end(), [](char character) {
return (character >= '0' && character <= '9') ||
(character >= 'a' && character <= 'f');
});
}

std::uint32_t rotate_right(std::uint32_t value, unsigned shift) noexcept {
return (value >> shift) | (value << (32U - shift));
}

int base64_value(char character) noexcept {
if (character >= 'A' && character <= 'Z') return character - 'A';
if (character >= 'a' && character <= 'z') return character - 'a' + 26;
if (character >= '0' && character <= '9') return character - '0' + 52;
if (character == '+') return 62;
if (character == '/') return 63;
return -1;
}

std::vector<std::uint8_t> decode_base64(std::string_view encoded) {
if (encoded.empty()) return {};
if (encoded.size() % 4U != 0U) {
throw std::invalid_argument("model base64 length");
}
std::vector<std::uint8_t> result;
result.reserve(encoded.size() / 4U * 3U);
for (std::size_t offset = 0; offset < encoded.size(); offset += 4U) {
const bool last = offset + 4U == encoded.size();
const char c0 = encoded[offset];
const char c1 = encoded[offset + 1U];
const char c2 = encoded[offset + 2U];
const char c3 = encoded[offset + 3U];
const int a = base64_value(c0);
const int b = base64_value(c1);
const int c = c2 == '=' ? 0 : base64_value(c2);
const int d = c3 == '=' ? 0 : base64_value(c3);
if (a < 0 || b < 0 || c < 0 || d < 0 || c0 == '=' || c1 == '=' ||
(!last && (c2 == '=' || c3 == '=')) ||
(c2 == '=' && c3 != '=')) {
throw std::invalid_argument("model base64 encoding");
}
result.push_back(static_cast<std::uint8_t>((a << 2U) | (b >> 4U)));
if (c2 == '=') {
if ((b & 15) != 0) throw std::invalid_argument("model base64 padding");
continue;
}
result.push_back(static_cast<std::uint8_t>((b << 4U) | (c >> 2U)));
if (c3 == '=') {
if ((c & 3) != 0) throw std::invalid_argument("model base64 padding");
continue;
}
result.push_back(static_cast<std::uint8_t>((c << 6U) | d));
}
return result;
}

class SplitMix64 {
public:
explicit SplitMix64(std::uint64_t state) noexcept : state_(state) {}
std::uint64_t next() noexcept {
state_ += 0x9e3779b97f4a7c15ULL;
return mix64(state_);
}
std::size_t index(std::size_t bound) noexcept {
if (bound < 2U) return 0U;
const std::uint64_t threshold = static_cast<std::uint64_t>(-bound) % bound;
for (;;) {
const std::uint64_t value = next();
if (value >= threshold) return static_cast<std::size_t>(value % bound);
}
}

private:
std::uint64_t state_{};
};

std::uint8_t canonical_direction(std::uint8_t direction,
std::uint8_t mover) noexcept {
return mover == 0 ? direction
: static_cast<std::uint8_t>((direction + 4U) & 7U);
}

bool canonical_action_less(const Action &left, const Action &right,
std::uint8_t mover) noexcept {
const std::size_t common = std::min(left.length, right.length);
for (std::size_t index = 0; index < common; ++index) {
const auto a = canonical_direction(left.directions[index], mover);
const auto b = canonical_direction(right.directions[index], mover);
if (a != b) return a < b;
}
return left.length < right.length;
}

void sort_canonical_arcs(std::array<Topology::Arc, 8> &arcs,
std::uint8_t count, std::uint8_t mover) noexcept {
const std::size_t limit = std::min<std::size_t>(count, arcs.size());
for (std::size_t index = 1; index < limit; ++index) {
const Topology::Arc value = arcs[index];
std::size_t position = index;
while (position != 0 &&
canonical_direction(value.direction, mover) <
canonical_direction(arcs[position - 1U].direction, mover)) {
arcs[position] = arcs[position - 1U];
--position;
}
arcs[position] = value;
}
}

std::uint8_t ordered_arcs(const State &state, std::uint64_t seed,
std::array<Topology::Arc, 8> &arcs) noexcept {
const std::uint8_t count = legal_arcs(state, arcs);
sort_canonical_arcs(arcs, count, state.to_move);
SplitMix64 random(seed ^ canonical_state_hash(state));
for (std::size_t remaining = count; remaining > 1U; --remaining) {
std::swap(arcs[remaining - 1U], arcs[random.index(remaining)]);
}
return count;
}

struct TurnTactics {
bool can_score{};
bool can_handoff{};
};

TurnTactics analyze_turn(const State &state) {
if (state.terminal()) return {};
const auto &topology = Topology::get();
const std::uint8_t mover = state.to_move;
std::array<bool, kVertexCount> seen{};
std::array<std::uint8_t, kVertexCount> queue{};
std::size_t head = 0;
std::size_t tail = 0;
seen[state.ball] = true;
queue[tail++] = state.ball;
TurnTactics result;
while (head < tail) {
const int vertex = queue[head++];
for (int index = 0; index < topology.degree(vertex); ++index) {
const auto arc = topology.arc(vertex, index);
if (edge_used(state, arc.edge)) continue;
if (topology.north_goal(arc.destination) ||
topology.south_goal(arc.destination)) {
const bool attacking = mover == 0
? topology.north_goal(arc.destination)
: topology.south_goal(arc.destination);
result.can_score = result.can_score || attacking;
continue;
}
const bool rebound = vertex_visited(state, arc.destination);
if (rebound) {
if (!seen[arc.destination]) {
seen[arc.destination] = true;
queue[tail++] = arc.destination;
}
continue;
}
bool reply = false;
for (int other = 0; other < topology.degree(arc.destination); ++other) {
const auto candidate = topology.arc(arc.destination, other);
if (candidate.edge != arc.edge && !edge_used(state, candidate.edge)) {
reply = true;
break;
}
}
result.can_handoff = result.can_handoff || reply;
}
}
return result;
}

std::optional<Action> exact_goal_path(const State &state, int goal_owner) {
if (state.terminal()) return std::nullopt;
const auto &topology = Topology::get();
constexpr int kUnseen = -2;
std::array<int, kVertexCount> parent{};
std::array<std::uint8_t, kVertexCount> parent_direction{};
std::array<std::uint8_t, kVertexCount> queue{};
parent.fill(kUnseen);
std::size_t head = 0;
std::size_t tail = 0;
parent[state.ball] = -1;
queue[tail++] = state.ball;
while (head < tail) {
const int vertex = queue[head++];
std::array<Topology::Arc, 8> arcs{};
std::uint8_t count = 0;
for (int index = 0; index < topology.degree(vertex); ++index) {
const auto arc = topology.arc(vertex, index);
if (!edge_used(state, arc.edge) && count < arcs.size()) arcs[count++] = arc;
}
sort_canonical_arcs(arcs, count, state.to_move);
for (std::uint8_t index = 0; index < count; ++index) {
const auto arc = arcs[index];
const bool is_goal = topology.north_goal(arc.destination) ||
topology.south_goal(arc.destination);
if (is_goal) {
const int owner = topology.north_goal(arc.destination) ? 0 : 1;
if (owner != goal_owner) continue;
parent[arc.destination] = vertex;
parent_direction[arc.destination] = arc.direction;
std::array<std::uint8_t, kMaximumActionLength> reverse{};
std::size_t length = 0;
for (int current = arc.destination; current != state.ball;
current = parent[current]) {
reverse[length++] = parent_direction[current];
}
Action action;
while (length != 0) action.directions[action.length++] = reverse[--length];
return action;
}
if (vertex_visited(state, arc.destination) &&
parent[arc.destination] == kUnseen) {
parent[arc.destination] = vertex;
parent_direction[arc.destination] = arc.direction;
queue[tail++] = arc.destination;
}
}
}
return std::nullopt;
}

TacticalClass classify(const State &result, std::uint8_t mover) {
if (result.terminal()) {
return result.winner == static_cast<int>(mover)
? TacticalClass::ImmediateGoal
: TacticalClass::OwnGoal;
}
const TurnTactics next = analyze_turn(result);
if (next.can_score) return TacticalClass::OpponentImmediateGoal;
if (!next.can_handoff) return TacticalClass::ForcedCutoff;
return TacticalClass::SafeHandoff;
}

class TurnGenerator {
public:
TurnGenerator(const State &root, GeneratorConfig config)
: root_(root), config_(config) {
if (root_.terminal() || config_.max_actions == 0 ||
config_.max_actions > kMaximumActions ||
config_.max_partial_paths == 0 ||
config_.max_partial_paths > kMaximumPartialPaths) {
throw std::invalid_argument("complete-turn generator config");
}
}

GenerationResult run() {
output_.actions.reserve(config_.max_actions);
retain_goal(root_.to_move);
retain_goal(1 - root_.to_move);
retain_witnesses();
std::deque<Partial> pending;
pending.push_back(Partial{root_, {}});
bool truncated = false;
while (!pending.empty()) {
if (output_.stats.partial_paths >= config_.max_partial_paths) {
truncated = true;
break;
}
if (expired()) {
output_.stats.deadline_reached = true;
truncated = true;
break;
}
++output_.stats.partial_paths;
Partial current;
if (output_.stats.partial_paths % 10U == 0U) {
++output_.stats.fifo_extractions;
current = std::move(pending.front());
pending.pop_front();
} else {
++output_.stats.lifo_extractions;
current = std::move(pending.back());
pending.pop_back();
}
std::array<Topology::Arc, 8> arcs{};
const std::uint8_t count =
ordered_arcs(current.state, config_.shuffle_seed, arcs);
for (std::uint8_t index = 0; index < count; ++index) {
Partial child = current;
if (!append(child.action, arcs[index].direction) ||
!apply_edge(child.state, arcs[index].direction)) {
continue;
}
if (child.state.terminal() || child.state.to_move != root_.to_move) {
retain(std::move(child));
} else if (pending.size() < config_.max_partial_paths) {
pending.push_back(std::move(child));
} else {
truncated = true;
}
}
}
if (output_.actions.empty()) {
const Action fallback_action = config_.precomputed_emergency == nullptr
? emergency_complete_action(root_, config_.shuffle_seed)
: *config_.precomputed_emergency;
Partial fallback{root_, fallback_action};
fallback.state = root_;
if (fallback.action.length == 0 ||
!apply_action(fallback.state, fallback.action)) {
throw std::logic_error("complete-turn emergency action");
}
retain(std::move(fallback));
truncated = true;
}
std::stable_sort(output_.actions.begin(), output_.actions.end(),
[](const auto &left, const auto &right) {
if (left.tactical != right.tactical) {
return left.tactical < right.tactical;
}
return left.order < right.order;
});
truncated = truncated || cap_dropped_;
output_.exhaustive = pending.empty() && !truncated;
if (truncated) ++output_.stats.truncations;
return std::move(output_);
}

private:
struct Partial {
State state{};
Action action{};
};

State root_{};
GeneratorConfig config_{};
GenerationResult output_{};
std::uint32_t next_order_{};
bool cap_dropped_{};

bool expired() const noexcept {
return config_.deadline != std::chrono::steady_clock::time_point::max() &&
std::chrono::steady_clock::now() >= config_.deadline;
}

void retain(Partial partial, bool count = true) {
if (count) ++output_.stats.completed_actions;
for (CompleteTurnAction &existing : output_.actions) {
if (!same_boundary(existing.result, partial.state)) continue;
++output_.stats.duplicate_boundaries;
if (canonical_action_less(partial.action, existing.action,
root_.to_move)) {
existing.action = partial.action;
}
return;
}
const TacticalClass tactical = classify(partial.state, root_.to_move);
const std::uint32_t order = next_order_++;
if (tactical != TacticalClass::SafeHandoff) {
++output_.stats.tactical_actions;
}
if (output_.actions.size() < config_.max_actions) {
output_.actions.push_back(
CompleteTurnAction{partial.state, partial.action, tactical, order});
return;
}
std::size_t worst = 0;
for (std::size_t index = 1; index < output_.actions.size(); ++index) {
const auto &left = output_.actions[index];
const auto &right = output_.actions[worst];
if (left.tactical > right.tactical ||
(left.tactical == right.tactical && left.order > right.order)) {
worst = index;
}
}
const TacticalClass retained_tactical = output_.actions[worst].tactical;
const std::uint32_t retained_order = output_.actions[worst].order;
if (tactical < retained_tactical ||
(tactical == retained_tactical && order < retained_order)) {
output_.actions[worst] =
CompleteTurnAction{partial.state, partial.action, tactical, order};
}
cap_dropped_ = true;
++output_.stats.cap_drops;
}

void retain_goal(int owner) {
if (expired()) return;
const auto path = exact_goal_path(root_, owner);
if (!path.has_value()) return;
Partial partial{root_, *path};
if (!apply_action(partial.state, partial.action)) {
throw std::logic_error("exact goal witness");
}
retain(std::move(partial), false);
}

void retain_witnesses() {
std::array<std::optional<Partial>, 3> best{};
std::deque<Partial> pending;
pending.push_back(Partial{root_, {}});
while (!pending.empty() && output_.stats.proof_paths < kProofPaths) {
if (expired()) {
output_.stats.proof_truncated = true;
break;
}
++output_.stats.proof_paths;
Partial current = std::move(pending.front());
pending.pop_front();
std::array<Topology::Arc, 8> arcs{};
const std::uint8_t count =
ordered_arcs(current.state, config_.shuffle_seed, arcs);
for (std::uint8_t index = 0; index < count; ++index) {
Partial child = current;
if (!append(child.action, arcs[index].direction) ||
!apply_edge(child.state, arcs[index].direction)) {
continue;
}
if (child.state.terminal() || child.state.to_move != root_.to_move) {
const int slot = static_cast<int>(classify(child.state, root_.to_move)) -
static_cast<int>(TacticalClass::ForcedCutoff);
if (slot < 0 || slot >= static_cast<int>(best.size())) continue;
auto &selected = best[static_cast<std::size_t>(slot)];
if (!selected.has_value() ||
canonical_action_less(child.action, selected->action,
root_.to_move)) {
selected = std::move(child);
}
} else if (pending.size() < kProofPaths) {
pending.push_back(std::move(child));
} else {
output_.stats.proof_truncated = true;
}
}
}
if (!pending.empty()) output_.stats.proof_truncated = true;
for (auto &witness : best) {
if (witness.has_value()) retain(std::move(*witness), false);
}
}
};

class BfmSearch {
public:
BfmSearch(const State &state, std::chrono::steady_clock::time_point deadline,
SearchConfig config, Action emergency)
: root_(state), deadline_(deadline), config_(config),
model_(config.model == nullptr ? deployment_model() : *config.model),
emergency_(emergency) {
if (root_.terminal() || config_.max_actions == 0 ||
config_.max_actions > kMaximumActions ||
config_.root_partial_paths == 0 ||
config_.root_partial_paths > kMaximumPartialPaths ||
config_.nonroot_partial_paths == 0 ||
config_.nonroot_partial_paths > kMaximumPartialPaths ||
config_.max_tree_nodes < 2 ||
config_.max_tree_nodes > kMaximumTreeNodes ||
config_.max_expansions == 0 ||
config_.max_expansions > kMaximumExpansions ||
!std::isfinite(config_.exploration) || config_.exploration < 0.0 ||
!std::isfinite(config_.fpu) ||
!std::isfinite(config_.final_visit_weight) ||
config_.final_visit_weight < 0.0 || emergency_.length == 0) {
throw std::invalid_argument("BFM search config");
}
nodes_.reserve(config_.max_tree_nodes);
}

SearchResult run() {
const float prior = model_.evaluate(active_features(root_));
nodes_.push_back(Node{root_, -1, {}, root_.to_move, prior, prior, 1, 0,
0, false, false, false, false, 0});
if (!expand(0)) throw std::logic_error("BFM root expansion");
refresh(0);
while (!nodes_[0].closed && budget_available()) {
const auto path = select_path();
if (!path.has_value()) break;
for (const int index : *path) {
++nodes_[index].visits;
++nodes_[index].selection_visits;
}
if (!expand(path->back())) break;
for (auto iterator = path->rbegin(); iterator != path->rend(); ++iterator) {
refresh(*iterator);
}
}
return result();
}

private:
struct Node {
State state{};
int parent{-1};
std::vector<int> children{};
std::uint8_t perspective{};
float value{};
float prior{};
std::uint32_t visits{1};
std::uint32_t selection_visits{};
std::uint32_t depth{};
bool expanded{};
bool solved{};
bool exhaustive{};
bool closed{};
std::uint32_t order{};
};
struct RootTranscript {
int child{};
Action action{};
TacticalClass tactical{TacticalClass::SafeHandoff};
};

State root_{};
std::chrono::steady_clock::time_point deadline_{};
SearchConfig config_{};
const QuantizedModel &model_;
Action emergency_{};
std::vector<Node> nodes_{};
std::vector<RootTranscript> roots_{};
SearchStats stats_{};

bool expired() {
if (deadline_ != std::chrono::steady_clock::time_point::max() &&
std::chrono::steady_clock::now() >= deadline_) {
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
return !expired();
}

void merge(const GeneratorStats &source) noexcept {
stats_.generator_partial_paths += source.partial_paths;
stats_.generator_proof_paths += source.proof_paths;
stats_.duplicate_boundaries += source.duplicate_boundaries;
stats_.fifo_extractions += source.fifo_extractions;
stats_.lifo_extractions += source.lifo_extractions;
stats_.deadline_reached = stats_.deadline_reached || source.deadline_reached;
}

bool expand(int index) {
if (nodes_[index].expanded || nodes_[index].solved) return true;
if (stats_.expansions >= config_.max_expansions ||
nodes_.size() >= config_.max_tree_nodes) {
return false;
}
GeneratorConfig generator;
generator.max_actions = std::min(
config_.max_actions, config_.max_tree_nodes - nodes_.size());
generator.max_partial_paths = index == 0
? config_.root_partial_paths : config_.nonroot_partial_paths;
generator.deadline = deadline_;
generator.shuffle_seed = config_.shuffle_seed;
generator.precomputed_emergency = index == 0 ? &emergency_ : nullptr;
GenerationResult generated = generate_complete_turns(nodes_[index].state,
generator);
merge(generated.stats);
if (generated.actions.empty() && index == 0) {
State fallback = root_;
if (!apply_action(fallback, emergency_)) return false;
generated.actions.push_back(CompleteTurnAction{
fallback, emergency_, classify(fallback, root_.to_move), 0});
generated.exhaustive = false;
}
if (generated.actions.empty()) return false;
const std::uint8_t child_perspective = 1U - nodes_[index].perspective;
const PreparedEvaluation prepared = model_.prepare(
active_features(nodes_[index].state, child_perspective));
const std::uint32_t depth = nodes_[index].depth + 1U;
stats_.max_depth = std::max(stats_.max_depth, depth);
const std::size_t first_child = nodes_.size();
for (CompleteTurnAction &action : generated.actions) {
float value{};
bool solved = false;
if (action.result.terminal()) {
value = mate_value(action.result.winner, child_perspective, depth);
solved = true;
} else if (action.tactical == TacticalClass::ForcedCutoff) {
value = mate_value(1 - child_perspective, child_perspective, depth + 1U);
solved = true;
} else if (action.tactical == TacticalClass::OpponentImmediateGoal) {
value = mate_value(child_perspective, child_perspective, depth + 1U);
solved = true;
} else {
value = model_.evaluate_delta(
prepared, active_features(action.result, child_perspective));
++stats_.evaluated_children;
}
nodes_.push_back(Node{action.result, index, {}, child_perspective,
value, value, 1, 0, depth, false, solved, false,
solved, action.order});
const int child = static_cast<int>(nodes_.size() - 1U);
if (index == 0) roots_.push_back(
RootTranscript{child, action.action, action.tactical});
++stats_.generated_children;
if (solved) ++stats_.tactical_children;
}
nodes_[index].children.reserve(nodes_.size() - first_child);
for (std::size_t child = first_child; child < nodes_.size(); ++child) {
nodes_[index].children.push_back(static_cast<int>(child));
}
nodes_[index].expanded = true;
nodes_[index].exhaustive = generated.exhaustive;
++stats_.expansions;
stats_.tree_nodes = nodes_.size();
refresh(index);
return true;
}

void refresh(int index) {
Node &node = nodes_[index];
if (node.children.empty()) return;
float best = -std::numeric_limits<float>::infinity();
bool proven_win = false;
bool all_solved = node.exhaustive;
bool all_closed = true;
for (const int child_index : node.children) {
const Node &child = nodes_[child_index];
const float action_value = -child.value;
best = std::max(best, action_value);
proven_win = proven_win || (child.solved && action_value > 0.0F);
all_solved = all_solved && child.solved;
all_closed = all_closed && child.closed;
}
if (!node.exhaustive) best = std::max(best, node.prior);
node.value = best;
node.solved = proven_win || all_solved;
node.closed = node.solved || (node.exhaustive && all_closed);
}

bool select_descendant(int current, std::vector<int> &path) const {
const Node &parent = nodes_[current];
if (parent.closed) return false;
if (!parent.expanded) return true;

struct Choice {
int child{};
double score{};
std::uint32_t order{};
};
std::vector<Choice> choices;
choices.reserve(parent.children.size());
for (const int child_index : parent.children) {
const Node &child = nodes_[child_index];
if (child.closed) continue;
choices.push_back(Choice{
child_index,
uct_score(-child.value, parent.selection_visits,
child.selection_visits, config_.exploration, config_.fpu),
child.order});
}
std::sort(choices.begin(), choices.end(), [](const Choice &left,
const Choice &right) {
if (left.score != right.score) return left.score > right.score;
return left.order < right.order;
});
for (const Choice &choice : choices) {
path.push_back(choice.child);
if (select_descendant(choice.child, path)) return true;
path.pop_back();
}
return false;
}

std::optional<std::vector<int>> select_path() const {
std::vector<int> path{0};
if (!select_descendant(0, path) || path.size() == 1U) {
return std::nullopt;
}
return path;
}

SearchResult result() const {
if (roots_.empty()) throw std::logic_error("BFM root action");
const RootTranscript *best = &roots_.front();
double best_score = -std::numeric_limits<double>::infinity();
SearchResult result;
result.root_actions.reserve(roots_.size());
for (const RootTranscript &root : roots_) {
const Node &child = nodes_[root.child];
const float value = -child.value;
const double score = final_score(value, child.visits,
config_.final_visit_weight);
if (score > best_score ||
(score == best_score &&
child.order < nodes_[best->child].order)) {
best = &root;
best_score = score;
}
result.root_actions.push_back(RootActionStat{
root.action, root.tactical, value, -child.prior, child.visits,
child.selection_visits, child.solved, child.order});
}
result.action = best->action;
result.value = -nodes_[best->child].value;
result.solved = nodes_[0].solved;
result.stats = stats_;
return result;
}
};

}  // namespace

bool Action::operator==(const Action &other) const noexcept {
return length == other.length &&
std::equal(directions.begin(), directions.begin() + length,
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
int vertex = 0;
for (int y = 1; y <= 11; ++y) {
for (int x = 0; x <= 8; ++x) {
xs_[vertex] = static_cast<std::int8_t>(x);
ys_[vertex++] = static_cast<std::int8_t>(y);
}
}
for (int x = 3; x <= 5; ++x) {
xs_[vertex] = static_cast<std::int8_t>(x);
ys_[vertex++] = 0;
xs_[vertex] = static_cast<std::int8_t>(x);
ys_[vertex++] = 12;
}
if (vertex != kVertexCount) throw std::logic_error("vertex topology");
for (int index = 0; index < kVertexCount; ++index) {
boundaries_[index] = boundary_coordinate(xs_[index], ys_[index]);
}
std::array<std::array<int, kVertexCount>, kVertexCount> edge_ids{};
for (auto &row : edge_ids) row.fill(-1);
int edge_count = 0;
for (int source = 0; source < 99; ++source) {
const int sx = xs_[source];
const int sy = ys_[source];
auto add_arc = [&](int destination) {
if (destination < 0 ||
!permitted_edge(sx, sy, xs_[destination], ys_[destination])) return;
int &edge = edge_ids[source][destination];
if (edge < 0) {
int &reverse = edge_ids[destination][source];
edge = reverse = edge_count++;
}
if (degrees_[source] >= 8) throw std::logic_error("topology degree");
const int direction = direction_for(xs_[destination] - sx,
ys_[destination] - sy);
if (direction < 0) throw std::logic_error("topology direction");
arcs_[source][degrees_[source]++] = Arc{
static_cast<std::uint16_t>(edge),
static_cast<std::uint8_t>(destination),
static_cast<std::uint8_t>(direction)};
};
for (int dx = -1; dx <= 1; ++dx) {
for (int dy = -1; dy <= 1; ++dy) {
if (dx == 0 && dy == 0) continue;
if (regular(sx + dx, sy + dy)) {
add_arc(vertex_at(sx + dx, sy + dy));
}
}
}
if ((sy == 1 || sy == 11) && sx >= 3 && sx <= 5) {
const int goal_y = sy == 1 ? 0 : 12;
for (int goal_x = 3; goal_x <= 5; ++goal_x) {
if (std::abs(sx - goal_x) <= 1) {
add_arc(vertex_at(goal_x, goal_y));
}
}
}
}
if (edge_count != kEdgeCount) throw std::logic_error("edge topology");
for (int index = 0; index < kVertexCount; ++index) {
const int rotated = vertex_at(8 - xs_[index], 12 - ys_[index]);
if (rotated < 0) throw std::logic_error("rotated vertex");
rotated_vertices_[index] = static_cast<std::uint8_t>(rotated);
}
for (int first = 0; first < kVertexCount; ++first) {
for (int second = first + 1; second < kVertexCount; ++second) {
const int edge = edge_ids[first][second];
if (edge < 0) continue;
const int rotated = edge_ids[rotated_vertices_[first]]
[rotated_vertices_[second]];
if (rotated < 0) throw std::logic_error("rotated edge");
rotated_edges_[edge] = static_cast<std::uint16_t>(rotated);
}
}
}

int Topology::vertex_at(int x, int y) const noexcept {
if (regular(x, y)) return (y - 1) * 9 + x;
if ((y == 0 || y == 12) && x >= 3 && x <= 5) {
return 99 + (x - 3) * 2 + (y == 12 ? 1 : 0);
}
return -1;
}

State initial_state() {
State state;
state.ball = static_cast<std::uint8_t>(Topology::get().vertex_at(4, 6));
return state;
}

bool same_boundary(const State &left, const State &right) noexcept {
return left.used == right.used && left.ball == right.ball &&
left.to_move == right.to_move && left.winner == right.winner;
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

std::uint8_t legal_arcs(const State &state,
std::array<Topology::Arc, 8> &result) noexcept {
if (state.terminal()) return 0;
const auto &topology = Topology::get();
std::uint8_t count = 0;
for (int index = 0; index < topology.degree(state.ball); ++index) {
const auto arc = topology.arc(state.ball, index);
if (!edge_used(state, arc.edge)) result[count++] = arc;
}
return count;
}

bool apply_edge(State &state, std::uint8_t direction) noexcept {
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
set_edge(state, selected.edge);
state.ball = selected.destination;
++state.ply;
if (topology.north_goal(state.ball) || topology.south_goal(state.ball)) {
state.winner = static_cast<std::int8_t>(
topology.north_goal(state.ball) ? 0 : 1);
return true;
}
if (!rebound) state.to_move = static_cast<std::uint8_t>(1 - mover);
std::array<Topology::Arc, 8> arcs{};
if (legal_arcs(state, arcs) == 0) {
state.winner = static_cast<std::int8_t>(1 - mover);
}
return true;
}

bool apply_action(State &state, const Action &action,
bool require_complete) noexcept {
if (state.terminal() || action.length == 0 ||
action.length > action.directions.size()) return false;
State next = state;
const std::uint8_t mover = next.to_move;
for (std::size_t index = 0; index < action.length; ++index) {
if (next.terminal() || next.to_move != mover ||
!apply_edge(next, action.directions[index])) return false;
if (index + 1U < action.length &&
(next.terminal() || next.to_move != mover)) return false;
}
if (require_complete && !next.terminal() && next.to_move == mover) return false;
state = next;
return true;
}

State rotate_and_swap(const State &state) {
State result = state;
result.used.fill(0);
const auto &topology = Topology::get();
for (int edge = 0; edge < kEdgeCount; ++edge) {
if (edge_used(state, edge)) set_edge(result, topology.rotated_edge(edge));
}
result.ball = static_cast<std::uint8_t>(topology.rotated_vertex(state.ball));
result.to_move = static_cast<std::uint8_t>(1 - state.to_move);
if (state.winner >= 0) result.winner = static_cast<std::int8_t>(1 - state.winner);
return result;
}

std::uint64_t state_hash(const State &state) noexcept {
std::uint64_t hash = mix64(
state.ball | (static_cast<std::uint64_t>(state.to_move) << 8U) |
(static_cast<std::uint64_t>(state.winner + 1) << 16U));
for (std::size_t index = 0; index < state.used.size(); ++index) {
hash ^= mix64(state.used[index] + index * 0x9e3779b97f4a7c15ULL);
}
return hash;
}

std::uint64_t canonical_state_hash(const State &state) noexcept {
return state_hash(state.to_move == 0 ? state : rotate_and_swap(state));
}

SparseFeatures active_features(const State &state) {
return active_features(state, state.to_move);
}

SparseFeatures active_features(const State &state, std::uint8_t perspective) {
if (perspective > 1) throw std::invalid_argument("feature perspective");
SparseFeatures result;
const auto &topology = Topology::get();
const bool rotate = perspective == 1;
for (int canonical = 0; canonical < kEdgeCount; ++canonical) {
const int physical = rotate ? topology.rotated_edge(canonical) : canonical;
if (edge_used(state, physical)) {
result.indices[result.count++] = static_cast<std::uint16_t>(canonical);
}
}
constexpr int kUnreachable = 1'000'000;
std::array<int, kVertexCount> distance{};
distance.fill(kUnreachable);
std::deque<int> pending;
distance[state.ball] = 0;
pending.push_front(state.ball);
while (!pending.empty()) {
const int source = pending.front();
pending.pop_front();
for (int index = 0; index < topology.degree(source); ++index) {
const auto arc = topology.arc(source, index);
if (edge_used(state, arc.edge)) continue;
const bool rebound = vertex_visited(state, arc.destination) ||
topology.north_goal(arc.destination) ||
topology.south_goal(arc.destination);
const int candidate = distance[source] + (rebound ? 0 : 1);
if (candidate >= distance[arc.destination]) continue;
distance[arc.destination] = candidate;
if (rebound) pending.push_front(arc.destination);
else pending.push_back(arc.destination);
}
}
for (int canonical = 0; canonical < kVertexCount; ++canonical) {
const int physical = rotate ? topology.rotated_vertex(canonical) : canonical;
std::size_t category = 56;
if (distance[physical] < 7) {
int free_degree = 0;
for (int index = 0; index < topology.degree(physical); ++index) {
free_degree += !edge_used(state, topology.arc(physical, index).edge);
}
category = static_cast<std::size_t>(distance[physical]) * 8U +
static_cast<std::size_t>(
std::clamp(free_degree - 1, 0, 7));
}
result.indices[result.count++] = static_cast<std::uint16_t>(
kEdgeCount + canonical * kVertexCategories + category);
}
std::sort(result.indices.begin(), result.indices.begin() + result.count);
return result;
}

std::string sha256_hex(std::span<const std::uint8_t> input) {
constexpr std::array<std::uint32_t, 64> constants{{
0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U}};
std::vector<std::uint8_t> bytes(input.begin(), input.end());
const std::uint64_t bits = static_cast<std::uint64_t>(bytes.size()) * 8U;
bytes.push_back(0x80U);
while (bytes.size() % 64U != 56U) bytes.push_back(0U);
for (int shift = 56; shift >= 0; shift -= 8) {
bytes.push_back(static_cast<std::uint8_t>(bits >> shift));
}
std::array<std::uint32_t, 8> hash{{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U}};
for (std::size_t offset = 0; offset < bytes.size(); offset += 64U) {
std::array<std::uint32_t, 64> words{};
for (std::size_t index = 0; index < 16; ++index) {
const std::size_t byte = offset + index * 4U;
words[index] = (static_cast<std::uint32_t>(bytes[byte]) << 24U) |
(static_cast<std::uint32_t>(bytes[byte + 1U]) << 16U) |
(static_cast<std::uint32_t>(bytes[byte + 2U]) << 8U) |
bytes[byte + 3U];
}
for (std::size_t index = 16; index < 64; ++index) {
const std::uint32_t a = words[index - 15U];
const std::uint32_t b = words[index - 2U];
words[index] = words[index - 16U] +
(rotate_right(a,7U)^rotate_right(a,18U)^(a>>3U)) +
words[index - 7U] +
(rotate_right(b,17U)^rotate_right(b,19U)^(b>>10U));
}
std::uint32_t a=hash[0],b=hash[1],c=hash[2],d=hash[3];
std::uint32_t e=hash[4],f=hash[5],g=hash[6],h=hash[7];
for (std::size_t index = 0; index < 64; ++index) {
const std::uint32_t s1=rotate_right(e,6U)^rotate_right(e,11U)^rotate_right(e,25U);
const std::uint32_t t1=h+s1+((e&f)^(~e&g))+constants[index]+words[index];
const std::uint32_t s0=rotate_right(a,2U)^rotate_right(a,13U)^rotate_right(a,22U);
const std::uint32_t t2=s0+((a&b)^(a&c)^(b&c));
h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
}
hash[0]+=a;hash[1]+=b;hash[2]+=c;hash[3]+=d;
hash[4]+=e;hash[5]+=f;hash[6]+=g;hash[7]+=h;
}
constexpr std::string_view digits="0123456789abcdef";
std::string result(64,'0');
std::size_t cursor=0;
for (const std::uint32_t word:hash) {
for (int shift=28;shift>=0;shift-=4) result[cursor++]=digits[(word>>shift)&15U];
}
return result;
}

float first_activation(float value) noexcept {
return value < 0.0F ? 0.01F * value : value * value;
}

float second_activation(float value) noexcept {
return value < 0.0F ? 0.01F * value : value;
}

float fast_tanh(float value) noexcept {
if (value < -4.95F) return -1.0F;
if (value > 4.95F) return 1.0F;
const float square = value * value;
return value * (135135.0F + square * (17325.0F + square * (378.0F + square))) /
(135135.0F + square * (62370.0F + square * (3150.0F + 28.0F * square)));
}

QuantizedModel::QuantizedModel(ModelDescriptor descriptor)
: hidden_one_(descriptor.hidden_one), hidden_two_(descriptor.hidden_two),
scale_one_(descriptor.scale_one), scale_two_(descriptor.scale_two),
scale_three_(descriptor.scale_three) {
const bool architecture =
descriptor.inputs == kFeatureCount &&
((hidden_one_ == 8 && (hidden_two_ == 8 || hidden_two_ == 16)) ||
(hidden_one_ == 12 && hidden_two_ == 8));
if (!architecture || !std::isfinite(scale_one_) || scale_one_ <= 0.0F ||
!std::isfinite(scale_two_) || scale_two_ <= 0.0F ||
!std::isfinite(scale_three_) || scale_three_ <= 0.0F) {
throw std::invalid_argument("model metadata");
}
const std::size_t count = kFeatureCount * hidden_one_ +
hidden_one_ * hidden_two_ + hidden_two_;
const std::vector<std::uint8_t> bytes = decode_base64(descriptor.packed_base64);
if (bytes.empty() && descriptor.allow_empty_bootstrap) {
weights_.assign(count, 0);
payload_sha256_ = sha256_hex({});
if (!descriptor.packed_sha256.empty() &&
descriptor.packed_sha256 != payload_sha256_) {
throw std::invalid_argument("bootstrap payload hash");
}
return;
}
if (!valid_sha256(descriptor.packed_sha256) ||
bytes.size() != (count * 3U + 7U) / 8U) {
throw std::invalid_argument("model payload length");
}
payload_sha256_ = sha256_hex(bytes);
if (payload_sha256_ != descriptor.packed_sha256) {
throw std::invalid_argument("model payload hash");
}
const std::size_t tail = count * 3U % 8U;
if (tail != 0U && (bytes.back() >> tail) != 0U) {
throw std::invalid_argument("model payload padding");
}
weights_.resize(count);
for (std::size_t index = 0; index < count; ++index) {
const std::size_t bit = index * 3U;
std::uint16_t window = bytes[bit / 8U];
if (bit % 8U > 5U && bit / 8U + 1U < bytes.size()) {
window |= static_cast<std::uint16_t>(bytes[bit / 8U + 1U]) << 8U;
}
const int code = static_cast<int>((window >> (bit % 8U)) & 7U);
const int signed_value = (code & 4) != 0 ? code - 8 : code;
if (signed_value == -4) throw std::invalid_argument("model code 100");
weights_[index] = static_cast<std::int8_t>(signed_value);
}
}

void QuantizedModel::add_input(std::array<std::int32_t, 12> &first,
std::size_t input,
int multiplier) const noexcept {
const std::size_t offset = input * hidden_one_;
for (std::size_t hidden = 0; hidden < hidden_one_; ++hidden) {
first[hidden] += multiplier * static_cast<int>(weights_[offset + hidden]);
}
}

PreparedEvaluation QuantizedModel::prepare(
const SparseFeatures &features) const noexcept {
PreparedEvaluation result;
result.features = features;
for (std::size_t index = 0; index < features.count; ++index) {
add_input(result.first, features.indices[index], 1);
}
return result;
}

float QuantizedModel::finish(std::array<std::int32_t, 12> first) const noexcept {
std::array<float, 12> activated{};
for (std::size_t input = 0; input < hidden_one_; ++input) {
activated[input] = first_activation(
static_cast<float>(first[input]) * scale_one_);
}
std::array<float, 16> second{};
const std::size_t offset_two = kFeatureCount * hidden_one_;
for (std::size_t input = 0; input < hidden_one_; ++input) {
for (std::size_t hidden = 0; hidden < hidden_two_; ++hidden) {
volatile float scaled = activated[input] * scale_two_;
volatile float term = scaled * static_cast<float>(
weights_[offset_two + input * hidden_two_ + hidden]);
second[hidden] = second[hidden] + term;
}
}
for (std::size_t hidden = 0; hidden < hidden_two_; ++hidden) {
second[hidden] = second_activation(second[hidden]);
}
const std::size_t offset_three = offset_two + hidden_one_ * hidden_two_;
float output = 0.0F;
for (std::size_t hidden = 0; hidden < hidden_two_; ++hidden) {
volatile float scaled = second[hidden] * scale_three_;
volatile float term = scaled * static_cast<float>(
weights_[offset_three + hidden]);
output = output + term;
}
return fast_tanh(output);
}

float QuantizedModel::evaluate(const SparseFeatures &features) const noexcept {
return finish(prepare(features).first);
}

float QuantizedModel::evaluate_delta(
const PreparedEvaluation &base,
const SparseFeatures &features) const noexcept {
auto first = base.first;
std::size_t left = 0;
std::size_t right = 0;
while (left < base.features.count || right < features.count) {
if (right == features.count ||
(left < base.features.count &&
base.features.indices[left] < features.indices[right])) {
add_input(first, base.features.indices[left++], -1);
} else if (left == base.features.count ||
features.indices[right] < base.features.indices[left]) {
add_input(first, features.indices[right++], 1);
} else {
++left;
++right;
}
}
return finish(first);
}

const QuantizedModel &deployment_model() {
static const QuantizedModel instance(ModelDescriptor{
model::kInputs, model::kHiddenOne, model::kHiddenTwo,
model::kScaleOne, model::kScaleTwo, model::kScaleThree,
model::kPackedWeights, model::kPayloadSha256, model::kBootstrapZero});
return instance;
}

Action emergency_complete_action(const State &source, std::uint64_t seed) {
Action action;
if (source.terminal()) return action;
State state = source;
const std::uint8_t mover = state.to_move;
while (!state.terminal() && state.to_move == mover) {
std::array<Topology::Arc, 8> arcs{};
const std::uint8_t count = ordered_arcs(state, seed, arcs);
if (count == 0 || !append(action, arcs[0].direction) ||
!apply_edge(state, arcs[0].direction)) return {};
}
return action;
}

GenerationResult generate_complete_turns(const State &state,
const GeneratorConfig &config) {
return TurnGenerator(state, config).run();
}

float mate_value(int winner, int perspective, std::uint32_t depth) noexcept {
const float magnitude =
std::max(1.0F, kMateScore - 10.0F * static_cast<float>(depth));
return winner == perspective ? magnitude : -magnitude;
}

double uct_score(float value, std::uint32_t parent_visits,
std::uint32_t child_visits, double exploration,
double fpu) {
if (!std::isfinite(value) || !std::isfinite(exploration) ||
exploration < 0.0 || !std::isfinite(fpu)) {
throw std::invalid_argument("UCT arguments");
}
if (child_visits == 0) return static_cast<double>(value) + fpu;
return static_cast<double>(value) + exploration *
std::sqrt(std::log(static_cast<double>(
std::max<std::uint32_t>(1, parent_visits))) /
static_cast<double>(child_visits));
}

double final_score(float value, std::uint32_t visits, double weight) {
if (!std::isfinite(value) || !std::isfinite(weight) || weight < 0.0) {
throw std::invalid_argument("final score arguments");
}
return static_cast<double>(value) + weight *
std::log(static_cast<double>(std::max<std::uint32_t>(1, visits)));
}

SearchResult search(const State &state,
std::chrono::steady_clock::time_point deadline,
const SearchConfig &config,
const Action *precomputed_emergency) {
const Action emergency = precomputed_emergency == nullptr
? emergency_complete_action(state, config.shuffle_seed)
: *precomputed_emergency;
return BfmSearch(state, deadline, config, emergency).run();
}

}  // namespace compact_value_bfm

namespace compact_value_bfm {
namespace {

bool parse_action(const std::string &text, Action &action) noexcept {
action = {};
if (text.empty() || text.size() > action.directions.size()) return false;
for (const char character : text) {
if (character < '0' || character > '7') return false;
action.directions[action.length++] =
static_cast<std::uint8_t>(character - '0');
}
return true;
}

[[maybe_unused]] int run_protocol() {
std::ios::sync_with_stdio(false);
std::cin.tie(nullptr);
int player_id = -1;
if (!(std::cin >> player_id) || (player_id != 0 && player_id != 1)) return 1;
State state = initial_state();
bool first_decision = true;
std::uint32_t decision = 0;
for (;;) {
int opponent_length = -1;
std::string opponent_text;
if (!(std::cin >> opponent_length >> opponent_text)) break;
if (opponent_text == "-") {
if (player_id != 0 || state.ply != 0 || opponent_length != 1) return 2;
} else {
if (opponent_length < 1 ||
static_cast<std::size_t>(opponent_length) != opponent_text.size() ||
state.to_move == player_id) return 3;
Action opponent;
if (!parse_action(opponent_text, opponent) ||
!apply_action(state, opponent)) return 4;
}
if (state.terminal()) break;
if (state.to_move != player_id) return 5;

const Action emergency = emergency_complete_action(state);
if (emergency.length == 0) return 6;
const int budget_ms = first_decision ? 800 : 155;
const auto started = std::chrono::steady_clock::now();
const auto deadline = started + std::chrono::milliseconds(budget_ms);
SearchResult result;
try {
result = search(state, deadline, SearchConfig{}, &emergency);
} catch (const std::exception &) {
result.action = emergency;
result.stats.deadline_reached = true;
}
if (result.action.length == 0 || !apply_action(state, result.action)) return 7;
std::cout << result.action.text() << '\n' << std::flush;
const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
std::chrono::steady_clock::now() - started).count();
std::cerr << "CVB d=" << decision
<< " n=" << result.stats.tree_nodes
<< " x=" << result.stats.expansions
<< " dl=" << result.stats.deadline_reached
<< " ms=" << elapsed
<< " v=" << result.value
<< " m=" << model::kIdentity << '\n';
++decision;
first_decision = false;
if (state.terminal()) break;
}
return 0;
}

}  // namespace
}  // namespace compact_value_bfm

#ifndef COMPACT_VALUE_BFM_NO_MAIN
int main() { return compact_value_bfm::run_protocol(); }
#endif
