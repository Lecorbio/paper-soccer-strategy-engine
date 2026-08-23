#include "jacek_replay_bfm/jacek_replay_bfm_internal.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "mcts_internal.hpp"
#include "papersoccer/geometry.hpp"

namespace papersoccer::detail {

static Point rotate_point(Point point) noexcept {
  return Point{8 - point.x, 12 - point.y};
}

static bool contains_arc(const SearchTopology::Adjacency &adjacency,
                         const SearchTopology::Arc &candidate) noexcept {
  for (std::uint8_t index = 0; index < adjacency.count; ++index) {
    const SearchTopology::Arc &arc = adjacency.arcs[index];
    if (arc.edge == candidate.edge &&
        arc.destination == candidate.destination) {
      return true;
    }
  }
  return false;
}

class ReplayBfmFeatureEncoder::Impl {
 public:
  explicit Impl(std::shared_ptr<const SearchTopology> topology)
      : topology_(std::move(topology)),
        cooperative_adjacency_(topology_->vertex_count()) {
    if (topology_->config().width != 8 ||
        topology_->config().height != 10 ||
        topology_->config().goal_rule != GoalRule::OwnGoalsAllowed ||
        topology_->config().blocked_rule != BlockedRule::MoverLoses ||
        topology_->edge_count() != kReplayBfmEdgeInputs ||
        topology_->vertex_count() != kReplayBfmVertices) {
      throw std::invalid_argument(
          "Jacek replay BFM features require 8x10 CodinGame rules");
    }
    initialize_maps();
  }

  ReplayBfmSparseFeatures encode(const SearchPosition &position,
                                 Player perspective) {
    calculate_distances(position);
    ReplayBfmSparseFeatures result;
    const bool rotate = perspective == Player::Two;

    for (std::size_t canonical = 0; canonical < kReplayBfmEdgeInputs;
         ++canonical) {
      const SearchTopology::EdgeIndex physical =
          rotate ? rotated_edges_[canonical]
                 : static_cast<SearchTopology::EdgeIndex>(canonical);
      if (position.edge_used(physical)) {
        result.indices[result.count++] =
            static_cast<std::uint16_t>(canonical);
      }
    }

    for (std::size_t canonical = 0; canonical < kReplayBfmVertices;
         ++canonical) {
      const SearchTopology::VertexIndex physical =
          rotate ? rotated_vertices_[canonical]
                 : static_cast<SearchTopology::VertexIndex>(canonical);
      const std::size_t category = category_for(position, physical);
      const std::size_t input =
          kReplayBfmEdgeInputs +
          canonical * kReplayBfmVertexCategories + category;
      result.indices[result.count++] = static_cast<std::uint16_t>(input);
    }
    return result;
  }

  PositionKey canonical_position_key(
      const SearchPosition &position) const noexcept {
    PositionKey result{};
    const bool rotate = position.to_move() == Player::Two;
    for (std::size_t canonical = 0; canonical < kReplayBfmEdgeInputs;
         ++canonical) {
      const SearchTopology::EdgeIndex physical =
          rotate ? rotated_edges_[canonical]
                 : static_cast<SearchTopology::EdgeIndex>(canonical);
      if (position.edge_used(physical)) {
        xor_position_key(result, position_key_component(1U, canonical));
      }
    }
    for (std::size_t canonical = 0; canonical < kReplayBfmVertices;
         ++canonical) {
      const SearchTopology::VertexIndex physical =
          rotate ? rotated_vertices_[canonical]
                 : static_cast<SearchTopology::VertexIndex>(canonical);
      if (position.vertex_visited(physical)) {
        xor_position_key(result, position_key_component(2U, canonical));
      }
    }
    const SearchTopology::VertexIndex canonical_ball =
        rotate ? rotated_vertices_[position.ball_vertex()]
               : position.ball_vertex();
    xor_position_key(result, position_key_component(3U, canonical_ball));
    // The mover is always Player One in the canonical frame.
    xor_position_key(
        result,
        position_key_component(4U, static_cast<std::uint64_t>(Player::One)));
    Status canonical_status = position.status();
    if (rotate && canonical_status == Status::WonByOne) {
      canonical_status = Status::WonByTwo;
    } else if (rotate && canonical_status == Status::WonByTwo) {
      canonical_status = Status::WonByOne;
    }
    xor_position_key(
        result,
        position_key_component(5U,
                               static_cast<std::uint64_t>(canonical_status)));
    return result;
  }

  std::uint8_t canonical_slot_rank(const SearchPosition &position,
                                   std::uint8_t slot) const noexcept {
    if (position.to_move() == Player::One) {
      return slot;
    }
    const SearchTopology::Adjacency &physical =
        topology_->adjacency(position.ball_vertex(), Player::Two);
    const SearchTopology::VertexIndex canonical_source =
        rotated_vertices_[position.ball_vertex()];
    const SearchTopology::VertexIndex canonical_destination =
        rotated_vertices_[physical.arcs[slot].destination];
    const SearchTopology::Adjacency &canonical =
        topology_->adjacency(canonical_source, Player::One);
    for (std::uint8_t rank = 0; rank < canonical.count; ++rank) {
      if (canonical.arcs[rank].destination == canonical_destination) {
        return rank;
      }
    }
    return slot;
  }

 private:
  std::shared_ptr<const SearchTopology> topology_;
  std::vector<SearchTopology::Adjacency> cooperative_adjacency_;
  std::array<SearchTopology::VertexIndex, kReplayBfmVertices>
      rotated_vertices_{};
  std::array<SearchTopology::EdgeIndex, kReplayBfmEdgeInputs>
      rotated_edges_{};
  std::array<int, kReplayBfmVertices> distances_{};
  std::deque<SearchTopology::VertexIndex> queue_{};

  void initialize_maps() {
    std::array<std::optional<Segment>, kReplayBfmEdgeInputs> segments{};
    for (SearchTopology::VertexIndex vertex = 0;
         vertex < topology_->vertex_count(); ++vertex) {
      SearchTopology::Adjacency &cooperative =
          cooperative_adjacency_[vertex];
      for (const Player player : {Player::One, Player::Two}) {
        const SearchTopology::Adjacency &source =
            topology_->adjacency(vertex, player);
        for (std::uint8_t slot = 0; slot < source.count; ++slot) {
          const SearchTopology::Arc &arc = source.arcs[slot];
          if (!contains_arc(cooperative, arc)) {
            if (cooperative.count >= cooperative.arcs.size()) {
              throw std::logic_error(
                  "Jacek replay BFM cooperative degree exceeds eight");
            }
            cooperative.arcs[cooperative.count++] = arc;
          }
          if (!segments[arc.edge].has_value()) {
            segments[arc.edge] =
                Segment{topology_->point(vertex),
                        topology_->point(arc.destination)};
          }
        }
      }
    }

    for (SearchTopology::VertexIndex vertex = 0;
         vertex < topology_->vertex_count(); ++vertex) {
      const auto rotated =
          topology_->find_vertex(rotate_point(topology_->point(vertex)));
      if (!rotated.has_value()) {
        throw std::logic_error(
            "Jacek replay BFM topology is not rotationally symmetric");
      }
      rotated_vertices_[vertex] = *rotated;
    }
    for (SearchTopology::EdgeIndex edge = 0;
         edge < topology_->edge_count(); ++edge) {
      if (!segments[edge].has_value()) {
        throw std::logic_error(
            "Jacek replay BFM topology contains an unobserved edge");
      }
      const Segment segment = *segments[edge];
      const auto rotated = topology_->find_edge(
          Segment{rotate_point(segment.a), rotate_point(segment.b)});
      if (!rotated.has_value()) {
        throw std::logic_error(
            "Jacek replay BFM edge map is not rotationally symmetric");
      }
      rotated_edges_[edge] = *rotated;
    }
  }

  void calculate_distances(const SearchPosition &position) {
    constexpr int unreachable = std::numeric_limits<int>::max() / 4;
    distances_.fill(unreachable);
    queue_.clear();
    distances_[position.ball_vertex()] = 0;
    queue_.push_front(position.ball_vertex());

    while (!queue_.empty()) {
      const SearchTopology::VertexIndex vertex = queue_.front();
      queue_.pop_front();
      const SearchTopology::Adjacency &adjacency =
          cooperative_adjacency_[vertex];
      for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
        const SearchTopology::Arc &arc = adjacency.arcs[slot];
        if (position.edge_used(arc.edge)) {
          continue;
        }
        const Point destination = topology_->point(arc.destination);
        const bool rebounds =
            position.vertex_visited(arc.destination) ||
            is_boundary_point(topology_->config(), destination) ||
            is_goal_point(topology_->config(), destination);
        const int candidate = distances_[vertex] + (rebounds ? 0 : 1);
        if (candidate >= distances_[arc.destination]) {
          continue;
        }
        distances_[arc.destination] = candidate;
        if (rebounds) {
          queue_.push_front(arc.destination);
        } else {
          queue_.push_back(arc.destination);
        }
      }
    }
  }

  std::size_t category_for(
      const SearchPosition &position,
      SearchTopology::VertexIndex vertex) const noexcept {
    const int distance = distances_[vertex];
    if (distance >= 7) {
      return 56;
    }

    const SearchTopology::Adjacency &adjacency =
        cooperative_adjacency_[vertex];
    std::size_t free_degree = 0;
    for (std::uint8_t slot = 0; slot < adjacency.count; ++slot) {
      free_degree += position.edge_used(adjacency.arcs[slot].edge) ? 0U : 1U;
    }
    const std::size_t degree_category =
        std::min<std::size_t>(free_degree == 0 ? 0 : free_degree - 1U, 7U);
    return static_cast<std::size_t>(distance) * 8U + degree_category;
  }
};

ReplayBfmFeatureEncoder::ReplayBfmFeatureEncoder(
    std::shared_ptr<const SearchTopology> topology)
    : impl_(std::make_unique<Impl>(std::move(topology))) {}

ReplayBfmFeatureEncoder::~ReplayBfmFeatureEncoder() = default;
ReplayBfmFeatureEncoder::ReplayBfmFeatureEncoder(
    ReplayBfmFeatureEncoder &&) noexcept = default;
ReplayBfmFeatureEncoder &ReplayBfmFeatureEncoder::operator=(
    ReplayBfmFeatureEncoder &&) noexcept = default;

ReplayBfmSparseFeatures ReplayBfmFeatureEncoder::encode(
    const SearchPosition &position) {
  return impl_->encode(position, position.to_move());
}

ReplayBfmSparseFeatures ReplayBfmFeatureEncoder::encode(
    const SearchPosition &position, Player perspective) {
  return impl_->encode(position, perspective);
}

PositionKey ReplayBfmFeatureEncoder::canonical_position_key(
    const SearchPosition &position) const noexcept {
  return impl_->canonical_position_key(position);
}

std::uint8_t ReplayBfmFeatureEncoder::canonical_slot_rank(
    const SearchPosition &position, std::uint8_t slot) const noexcept {
  return impl_->canonical_slot_rank(position, slot);
}

ReplayBfmSparseFeatures replay_bfm_sparse_features(const GameState &state) {
  auto topology = std::make_shared<SearchTopology>(state.config);
  SearchPosition position(topology, state);
  ReplayBfmFeatureEncoder encoder(topology);
  return encoder.encode(position);
}

}  // namespace papersoccer::detail
