#include "jacek_features.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>

#include "papersoccer/geometry.hpp"

namespace papersoccer::detail {

namespace {

Point rotate_point(const RulesConfig &config, Point point) noexcept {
  return Point{config.width - point.x, config.height + 2 - point.y};
}

bool contains_arc(const SearchTopology::Adjacency &adjacency,
                  const SearchTopology::Arc &candidate) noexcept {
  for (std::uint8_t index = 0; index < adjacency.count; ++index) {
    const SearchTopology::Arc &arc = adjacency.arcs[index];
    if (arc.destination == candidate.destination && arc.edge == candidate.edge) {
      return true;
    }
  }
  return false;
}

}  // namespace

JacekFeatureEncoder::JacekFeatureEncoder(
    std::shared_ptr<const SearchTopology> topology)
    : topology_(std::move(topology)) {
  if (!topology_) {
    throw std::invalid_argument("Jacek feature encoder requires a topology");
  }
  const RulesConfig &config = topology_->config();
  if (config.width != 8 || config.height != 10 ||
      topology_->vertex_count() != kJacekVertices ||
      topology_->edge_count() != kJacekEdgeInputs) {
    throw std::invalid_argument(
        "Jacek-inspired features require the standard 8x10 board");
  }

  cooperative_adjacency_.resize(topology_->vertex_count());
  std::array<std::optional<Segment>, kJacekEdgeInputs> edge_segments{};
  for (SearchTopology::VertexIndex vertex = 0;
       vertex < topology_->vertex_count(); ++vertex) {
    SearchTopology::Adjacency &cooperative = cooperative_adjacency_[vertex];
    for (const Player player : {Player::One, Player::Two}) {
      const SearchTopology::Adjacency &source =
          topology_->adjacency(vertex, player);
      for (std::uint8_t index = 0; index < source.count; ++index) {
        const SearchTopology::Arc &arc = source.arcs[index];
        if (!contains_arc(cooperative, arc)) {
          if (cooperative.count >= cooperative.arcs.size()) {
            throw std::logic_error(
                "Jacek cooperative topology exceeds eight arcs");
          }
          cooperative.arcs[cooperative.count++] = arc;
        }
        if (!edge_segments[arc.edge].has_value()) {
          edge_segments[arc.edge] =
              Segment{topology_->point(vertex),
                      topology_->point(arc.destination)};
        }
      }
    }
  }

  for (SearchTopology::VertexIndex vertex = 0;
       vertex < topology_->vertex_count(); ++vertex) {
    const auto rotated =
        topology_->find_vertex(rotate_point(config, topology_->point(vertex)));
    if (!rotated.has_value()) {
      throw std::logic_error("Jacek topology is not rotationally symmetric");
    }
    rotated_vertices_[vertex] = *rotated;
  }
  for (SearchTopology::EdgeIndex edge = 0;
       edge < topology_->edge_count(); ++edge) {
    if (!edge_segments[edge].has_value()) {
      throw std::logic_error("Jacek topology contains an unobserved edge");
    }
    const Segment segment = *edge_segments[edge];
    const auto rotated = topology_->find_edge(
        Segment{rotate_point(config, segment.a),
                rotate_point(config, segment.b)});
    if (!rotated.has_value()) {
      throw std::logic_error("Jacek edge topology is not rotationally symmetric");
    }
    rotated_edges_[edge] = *rotated;
  }
}

void JacekFeatureEncoder::calculate_turn_distances(
    const SearchPosition &position) {
  constexpr int kUnreachable = std::numeric_limits<int>::max() / 4;
  distances_.fill(kUnreachable);
  queue_.clear();
  distances_[position.ball_vertex()] = 0;
  queue_.push_back(position.ball_vertex());

  while (!queue_.empty()) {
    const SearchTopology::VertexIndex vertex = queue_.front();
    queue_.pop_front();
    const int source_distance = distances_[vertex];
    const SearchTopology::Adjacency &adjacency =
        cooperative_adjacency_[vertex];
    for (std::uint8_t index = 0; index < adjacency.count; ++index) {
      const SearchTopology::Arc &arc = adjacency.arcs[index];
      if (position.edge_used(arc.edge)) {
        continue;
      }
      const Point destination = topology_->point(arc.destination);
      const bool rebounds = position.vertex_visited(arc.destination) ||
                            is_boundary_point(topology_->config(), destination) ||
                            is_goal_point(topology_->config(), destination);
      const int candidate = source_distance + (rebounds ? 0 : 1);
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

JacekSparseFeatures JacekFeatureEncoder::sparse_features(
    const SearchPosition &position) {
  if (!same_rules_config(position.topology()->config(),
                         topology_->config())) {
    throw std::invalid_argument(
        "Jacek feature position does not match its topology");
  }
  calculate_turn_distances(position);

  JacekSparseFeatures result;
  const bool rotate = position.to_move() == Player::Two;
  for (std::size_t canonical_edge = 0;
       canonical_edge < kJacekEdgeInputs; ++canonical_edge) {
    const SearchTopology::EdgeIndex physical_edge =
        rotate ? rotated_edges_[canonical_edge]
               : static_cast<SearchTopology::EdgeIndex>(canonical_edge);
    if (position.edge_used(physical_edge)) {
      result.indices[result.count++] =
          static_cast<std::uint16_t>(canonical_edge);
    }
  }
  for (std::size_t canonical_vertex = 0;
       canonical_vertex < kJacekVertices; ++canonical_vertex) {
    const SearchTopology::VertexIndex physical_vertex =
        rotate ? rotated_vertices_[canonical_vertex]
               : static_cast<SearchTopology::VertexIndex>(canonical_vertex);
    const int distance =
        std::min(distances_[physical_vertex],
                 static_cast<int>(kJacekDistanceBuckets - 1));
    const std::size_t input =
        kJacekEdgeInputs +
        canonical_vertex * kJacekDistanceBuckets +
        static_cast<std::size_t>(distance);
    result.indices[result.count++] = static_cast<std::uint16_t>(input);
  }
  return result;
}

std::array<std::uint8_t, kJacekInputCount>
JacekFeatureEncoder::dense_features(const SearchPosition &position) {
  std::array<std::uint8_t, kJacekInputCount> result{};
  const JacekSparseFeatures sparse = sparse_features(position);
  for (std::size_t index = 0; index < sparse.count; ++index) {
    result[sparse.indices[index]] = 1;
  }
  return result;
}

JacekSparseFeatures jacek_article_sparse_features(const GameState &state) {
  auto topology = std::make_shared<SearchTopology>(state.config);
  SearchPosition position(topology, state);
  JacekFeatureEncoder encoder(topology);
  return encoder.sparse_features(position);
}

std::array<std::uint8_t, kJacekInputCount> jacek_article_dense_features(
    const GameState &state) {
  auto topology = std::make_shared<SearchTopology>(state.config);
  SearchPosition position(topology, state);
  JacekFeatureEncoder encoder(topology);
  return encoder.dense_features(position);
}

}  // namespace papersoccer::detail
