#pragma once

namespace papersoccer::neural_puct {

struct PrimitiveVisitTarget {
  std::array<float, 8> probabilities{};
  std::uint64_t total_visits{};
  bool fallback{};
};

class NeuralPuctTrainingAccess {
 public:
  static PrimitiveVisitTarget root(const NeuralPuctSearch &search) {
    const NeuralPuctSearch::Node &node = search.nodes_.front();
    if (!node.expanded || node.edge_count == 0) {
      throw std::logic_error("neural PUCT root was not expanded");
    }
    PrimitiveVisitTarget target;
    if (node.exact_winning_child.has_value()) {
      target.total_visits = 1;
      target.probabilities[
          node.edges[*node.exact_winning_child].canonical_direction] = 1.0F;
      return target;
    }
    for (std::uint8_t index = 0; index < node.edge_count; ++index) {
      target.total_visits += node.edges[index].visits;
    }
    if (target.total_visits != 0) {
      for (std::uint8_t index = 0; index < node.edge_count; ++index) {
        target.probabilities[node.edges[index].canonical_direction] =
            static_cast<float>(node.edges[index].visits) /
            static_cast<float>(target.total_visits);
      }
      return target;
    }
    const std::uint8_t selected = search.visit_max_edge(node);
    target.probabilities[node.edges[selected].canonical_direction] = 1.0F;
    target.fallback = true;
    return target;
  }

  static std::vector<PrimitiveVisitTarget> collect(
      NeuralPuctSearch &search, const std::vector<Move> &selected_action) {
    const std::size_t root_depth = search.position_.undo_depth();
    std::vector<PrimitiveVisitTarget> targets;
    targets.reserve(selected_action.size());
    std::optional<std::uint32_t> node_index = 0;
    try {
      for (const Move expected : selected_action) {
        if (search.position_.is_terminal() ||
            search.position_.to_move() != search.root_mover_) {
          throw std::logic_error("selected action continues after turn end");
        }
        std::uint8_t slot{};
        std::uint32_t child = NeuralPuctSearch::kNoNode;
        const NeuralPuctSearch::Node *node = nullptr;
        if (node_index.has_value() && search.nodes_[*node_index].expanded &&
            search.nodes_[*node_index].edge_count != 0) {
          node = &search.nodes_[*node_index];
          const std::uint8_t edge = search.visit_max_edge(*node);
          if (node->edges[edge].move != expected) {
            throw std::logic_error("selected action is not visit-max");
          }
          slot = node->edges[edge].slot;
          child = node->edges[edge].child;
        } else {
          std::array<std::uint8_t, detail::kMaximumMoves> slots{};
          const std::uint8_t count = search.position_.legal_slots(slots);
          bool found = false;
          for (std::uint8_t index = 0; index < count; ++index) {
            if (search.position_.move_for_slot(slots[index]) == expected) {
              slot = slots[index];
              found = true;
              break;
            }
          }
          if (!found) {
            throw std::logic_error("selected fallback action is not legal");
          }
        }

        PrimitiveVisitTarget target;
        if (node != nullptr && !node->exact_winning_child.has_value()) {
          for (std::uint8_t index = 0; index < node->edge_count; ++index) {
            target.total_visits += node->edges[index].visits;
          }
          if (target.total_visits != 0) {
            for (std::uint8_t index = 0; index < node->edge_count; ++index) {
              target.probabilities[node->edges[index].canonical_direction] =
                  static_cast<float>(node->edges[index].visits) /
                  static_cast<float>(target.total_visits);
            }
          }
        }
        if (target.total_visits == 0) {
          const std::size_t direction = canonical_direction(
              search.position_.ball(), expected.to,
              search.position_.to_move());
          target.probabilities[direction] = 1.0F;
          target.fallback = true;
        }
        targets.push_back(target);
        search.position_.make_move(slot);
        node_index = child == NeuralPuctSearch::kNoNode
                         ? std::nullopt
                         : std::optional<std::uint32_t>(child);
      }
      if (selected_action.empty() ||
          (!search.position_.is_terminal() &&
           search.position_.to_move() == search.root_mover_)) {
        throw std::logic_error("selected action is not a complete turn");
      }
    } catch (...) {
      search.position_.unmake_to(root_depth);
      throw;
    }
    search.position_.unmake_to(root_depth);
    return targets;
  }
};

}  // namespace papersoccer::neural_puct
