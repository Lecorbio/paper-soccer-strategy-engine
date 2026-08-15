#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// This is an audit harness, not production code. Exposing private members lets
// it test move ordering and the actual scalar evaluation without modifying the
// frozen Rank-4 implementation.
#define private public
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "../../submissions/codingame/bots/rank_4/submission.cpp"
#undef PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#undef private

namespace ps = papersoccer;
namespace rank4 = papersoccer::turn_action_v2;

namespace {

struct AuditCounts {
  std::size_t checked{};
  std::size_t mismatched{};
  std::string first_mismatch;
};

struct CorpusState {
  ps::GameState state;
  std::string transcript;
};

ps::RulesConfig codingame_rules() {
  return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                         ps::BlockedRule::MoverLoses};
}

ps::Point rotate_point(ps::Point point) {
  return {8 - point.x, 12 - point.y};
}

ps::Status swap_status(ps::Status status) {
  if (status == ps::Status::WonByOne) return ps::Status::WonByTwo;
  if (status == ps::Status::WonByTwo) return ps::Status::WonByOne;
  return status;
}

ps::GameState rotate_and_swap(const ps::GameState &state) {
  ps::GameState result = state;
  result.ball = rotate_point(state.ball);
  result.to_move = ps::opponent(state.to_move);
  result.status = swap_status(state.status);
  result.path.clear();
  for (const ps::Point point : state.path) {
    result.path.push_back(rotate_point(point));
  }
  result.used_segments.clear();
  for (const ps::Segment &edge : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{rotate_point(edge.a), rotate_point(edge.b)});
  }
  result.visit_count.clear();
  for (const auto &[point, count] : state.visit_count) {
    result.visit_count[rotate_point(point)] = count;
  }
  return result;
}

char rotate_direction(char direction) {
  return static_cast<char>('0' + (direction - '0' + 4) % 8);
}

std::string rotate_action(std::string_view action) {
  std::string result;
  result.reserve(action.size());
  for (const char direction : action) {
    result.push_back(rotate_direction(direction));
  }
  return result;
}

std::string encode_moves(ps::GameState state,
                         const std::vector<ps::Move> &moves) {
  std::string result;
  result.reserve(moves.size());
  for (const ps::Move move : moves) {
    result.push_back(rank4::encode_direction(state.ball, move.to));
    state = ps::apply_move(state, move);
  }
  return result;
}

std::string ordering_string(rank4::CompleteTurnSearch &search) {
  const ps::Point ball = search.position_.ball();
  const rank4::OrderedMoveList ordered = search.ordered_moves();
  std::ostringstream output;
  for (std::uint8_t index = 0; index < ordered.count; ++index) {
    if (index != 0) output << ',';
    output << rank4::encode_direction(ball, ordered.values[index].move.to)
           << ':' << ordered.values[index].score;
  }
  return output.str();
}

std::string rotated_ordering_string(rank4::CompleteTurnSearch &search) {
  const ps::Point ball = search.position_.ball();
  const rank4::OrderedMoveList ordered = search.ordered_moves();
  std::ostringstream output;
  for (std::uint8_t index = 0; index < ordered.count; ++index) {
    if (index != 0) output << ',';
    const char direction =
        rank4::encode_direction(ball, ordered.values[index].move.to);
    output << rotate_direction(direction) << ':'
           << ordered.values[index].score;
  }
  return output.str();
}

std::uint64_t next_random(std::uint64_t &state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

std::vector<CorpusState> make_corpus(std::size_t target) {
  std::vector<CorpusState> corpus;
  corpus.reserve(target);
  std::uint64_t random_state = 0xd1b54a32d192ed03ULL;
  while (corpus.size() < target) {
    ps::GameState state = ps::make_initial_state(codingame_rules());
    std::string transcript;
    for (std::size_t turn = 0;
         turn < 160 && !ps::is_terminal(state) && corpus.size() < target;
         ++turn) {
      corpus.push_back(CorpusState{state, transcript});
      const ps::Player mover = state.to_move;
      std::string action;
      while (!ps::is_terminal(state) && state.to_move == mover) {
        const std::vector<ps::Move> legal = ps::legal_moves(state);
        if (legal.empty()) {
          throw std::logic_error("in-progress corpus state has no legal move");
        }
        const ps::Move move =
            legal[next_random(random_state) % legal.size()];
        action.push_back(rank4::encode_direction(state.ball, move.to));
        state = ps::apply_move(state, move);
      }
      if (!transcript.empty()) transcript.push_back('/');
      transcript += action;
    }
  }
  return corpus;
}

void remember_mismatch(AuditCounts &counts, std::string detail) {
  ++counts.mismatched;
  if (counts.first_mismatch.empty()) {
    counts.first_mismatch = std::move(detail);
  }
}

void audit_evaluation(const std::vector<CorpusState> &corpus,
                      AuditCounts &snapshot_counts,
                      AuditCounts &feature_bit_counts,
                      AuditCounts &score_counts,
                      float &maximum_feature_delta) {
  rank4::SearchConfig config;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  for (const CorpusState &item : corpus) {
    const ps::GameState rotated = rotate_and_swap(item.state);
    rank4::CompleteTurnSearch original(item.state, config);
    rank4::CompleteTurnSearch transformed(rotated, config);
    const rank4::EvaluationSnapshot left = original.evaluation_snapshot();
    const rank4::EvaluationSnapshot right = transformed.evaluation_snapshot();
    ++snapshot_counts.checked;
    ++feature_bit_counts.checked;
    float local_delta = 0.0F;
    bool bit_mismatch = false;
    std::size_t first_bit_mismatch = left.features.size();
    for (std::size_t index = 0; index < left.features.size(); ++index) {
      local_delta = std::max(
          local_delta, std::abs(left.features[index] - right.features[index]));
      if (left.features[index] != right.features[index]) {
        bit_mismatch = true;
        if (first_bit_mismatch == left.features.size()) {
          first_bit_mismatch = index;
        }
      }
    }
    maximum_feature_delta = std::max(maximum_feature_delta, local_delta);
    if (left.anchor_score != -right.anchor_score ||
        left.mover_sign != -right.mover_sign || local_delta >= 1e-5F) {
      std::ostringstream detail;
      detail << "transcript=" << item.transcript
             << " anchor=" << left.anchor_score << '/' << right.anchor_score
             << " feature_delta=" << local_delta;
      remember_mismatch(snapshot_counts, detail.str());
    }
    if (bit_mismatch) {
      std::ostringstream detail;
      detail << "transcript=" << item.transcript
             << " first_feature=" << first_bit_mismatch << " values="
             << std::setprecision(9) << left.features[first_bit_mismatch]
             << '/' << right.features[first_bit_mismatch]
             << " max_delta=" << local_delta;
      remember_mismatch(feature_bit_counts, detail.str());
    }

    const int left_score = original.evaluate();
    const int right_score = transformed.evaluate();
    ++score_counts.checked;
    if (left_score != -right_score) {
      std::ostringstream detail;
      detail << "transcript=" << item.transcript << " score=" << left_score
             << '/' << right_score;
      remember_mismatch(score_counts, detail.str());
    }
  }
}

void audit_ordering(const std::vector<CorpusState> &corpus,
                    AuditCounts &counts) {
  rank4::SearchConfig config;
  config.transposition_entries = 2;
  config.evaluation_entries = 2;
  for (const CorpusState &item : corpus) {
    rank4::CompleteTurnSearch original(item.state, config);
    rank4::CompleteTurnSearch transformed(rotate_and_swap(item.state), config);
    const std::string left = ordering_string(original);
    const std::string right_canonical = rotated_ordering_string(transformed);
    ++counts.checked;
    if (left != right_canonical) {
      std::ostringstream detail;
      detail << "transcript=" << item.transcript << " original=" << left
             << " rotated_back=" << right_canonical;
      remember_mismatch(counts, detail.str());
    }
  }
}

void audit_fallback(const std::vector<CorpusState> &corpus,
                    AuditCounts &counts) {
  rank4::SearchConfig config;
  config.transposition_entries = 2;
  config.evaluation_entries = 2;
  for (const CorpusState &item : corpus) {
    const ps::GameState rotated = rotate_and_swap(item.state);
    rank4::CompleteTurnSearch original(item.state, config);
    rank4::CompleteTurnSearch transformed(rotated, config);
    const std::string left = encode_moves(item.state, original.fallback_action());
    const std::string right = encode_moves(rotated, transformed.fallback_action());
    ++counts.checked;
    if (rotate_action(left) != right) {
      std::ostringstream detail;
      detail << "transcript=" << item.transcript << " original=" << left
             << " rotated=" << right
             << " expected=" << rotate_action(left);
      remember_mismatch(counts, detail.str());
    }
  }
}

void audit_fixed_nodes(const std::vector<CorpusState> &corpus,
                       std::uint64_t nodes, AuditCounts &action_counts,
                       AuditCounts &score_counts, AuditCounts &stats_counts) {
  for (const CorpusState &item : corpus) {
    rank4::SearchConfig config;
    config.max_nodes = nodes;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent = 100;
    const ps::GameState rotated = rotate_and_swap(item.state);
    rank4::CompleteTurnSearch original(item.state, config);
    rank4::CompleteTurnSearch transformed(rotated, config);
    const std::string left = encode_moves(item.state, original.run());
    const std::string right = encode_moves(rotated, transformed.run());
    ++action_counts.checked;
    ++score_counts.checked;
    ++stats_counts.checked;
    if (rotate_action(left) != right) {
      std::ostringstream detail;
      detail << "nodes=" << nodes << " transcript=" << item.transcript
             << " original=" << left << " rotated=" << right
             << " expected=" << rotate_action(left)
             << " completed_depth=" << original.stats().completed_turn_depth
             << '/' << transformed.stats().completed_turn_depth;
      remember_mismatch(action_counts, detail.str());
    }
    if (original.stats().root_score != -transformed.stats().root_score) {
      std::ostringstream detail;
      detail << "nodes=" << nodes << " transcript=" << item.transcript
             << " score=" << original.stats().root_score << '/'
             << transformed.stats().root_score;
      remember_mismatch(score_counts, detail.str());
    }
    if (original.stats().nodes != transformed.stats().nodes ||
        original.stats().completed_turn_depth !=
            transformed.stats().completed_turn_depth ||
        original.stats().attempted_turn_depth !=
            transformed.stats().attempted_turn_depth ||
        original.stats().completed_actions !=
            transformed.stats().completed_actions ||
        original.stats().cutoffs != transformed.stats().cutoffs ||
        original.stats().transposition_hits !=
            transformed.stats().transposition_hits) {
      std::ostringstream detail;
      detail << "nodes=" << nodes << " transcript=" << item.transcript
             << " visited=" << original.stats().nodes << '/'
             << transformed.stats().nodes << " depth="
             << original.stats().completed_turn_depth << '/'
             << transformed.stats().completed_turn_depth << " actions="
             << original.stats().completed_actions << '/'
             << transformed.stats().completed_actions << " cutoffs="
             << original.stats().cutoffs << '/'
             << transformed.stats().cutoffs << " tt_hits="
             << original.stats().transposition_hits << '/'
             << transformed.stats().transposition_hits;
      remember_mismatch(stats_counts, detail.str());
    }
  }
}

void audit_fixed_depth(const std::vector<CorpusState> &corpus,
                       std::uint32_t depth, AuditCounts &action_counts,
                       AuditCounts &score_counts, AuditCounts &stats_counts,
                       AuditCounts &incomplete_counts) {
  for (const CorpusState &item : corpus) {
    rank4::SearchConfig config;
    config.max_turn_depth = depth;
    config.max_nodes = 2'000'000;
    config.replay_value_blend_percent = 15;
    config.teacher_residual_weight_percent = 100;
    const ps::GameState rotated = rotate_and_swap(item.state);
    rank4::CompleteTurnSearch original(item.state, config);
    rank4::CompleteTurnSearch transformed(rotated, config);
    const std::string left = encode_moves(item.state, original.run());
    const std::string right = encode_moves(rotated, transformed.run());
    if (original.stats().completed_turn_depth != depth ||
        transformed.stats().completed_turn_depth != depth) {
      ++incomplete_counts.checked;
      std::ostringstream detail;
      detail << "depth=" << depth << " transcript=" << item.transcript
             << " completed=" << original.stats().completed_turn_depth << '/'
             << transformed.stats().completed_turn_depth;
      remember_mismatch(incomplete_counts, detail.str());
      continue;
    }
    ++action_counts.checked;
    ++score_counts.checked;
    ++stats_counts.checked;
    if (rotate_action(left) != right) {
      std::ostringstream detail;
      detail << "depth=" << depth << " transcript=" << item.transcript
             << " original=" << left << " rotated=" << right
             << " expected=" << rotate_action(left);
      remember_mismatch(action_counts, detail.str());
    }
    if (original.stats().root_score != -transformed.stats().root_score) {
      std::ostringstream detail;
      detail << "depth=" << depth << " transcript=" << item.transcript
             << " score=" << original.stats().root_score << '/'
             << transformed.stats().root_score;
      remember_mismatch(score_counts, detail.str());
    }
    if (original.stats().nodes != transformed.stats().nodes ||
        original.stats().completed_actions !=
            transformed.stats().completed_actions ||
        original.stats().cutoffs != transformed.stats().cutoffs ||
        original.stats().transposition_hits !=
            transformed.stats().transposition_hits) {
      std::ostringstream detail;
      detail << "depth=" << depth << " transcript=" << item.transcript
             << " visited=" << original.stats().nodes << '/'
             << transformed.stats().nodes << " actions="
             << original.stats().completed_actions << '/'
             << transformed.stats().completed_actions << " cutoffs="
             << original.stats().cutoffs << '/'
             << transformed.stats().cutoffs << " tt_hits="
             << original.stats().transposition_hits << '/'
             << transformed.stats().transposition_hits;
      remember_mismatch(stats_counts, detail.str());
    }
  }
}

void print_counts(std::string_view label, const AuditCounts &counts) {
  std::cout << label << " checked=" << counts.checked
            << " mismatched=" << counts.mismatched << '\n';
  if (!counts.first_mismatch.empty()) {
    std::cout << "  first: " << counts.first_mismatch << '\n';
  }
}

}  // namespace

int main(int argc, char **argv) {
  const std::size_t corpus_size =
      argc >= 2 ? static_cast<std::size_t>(std::stoull(argv[1])) : 120U;
  const std::vector<CorpusState> corpus = make_corpus(corpus_size);

  AuditCounts snapshots;
  AuditCounts feature_bits;
  AuditCounts evaluations;
  AuditCounts ordering;
  AuditCounts fallback;
  float maximum_feature_delta = 0.0F;
  audit_evaluation(corpus, snapshots, feature_bits, evaluations,
                   maximum_feature_delta);
  audit_ordering(corpus, ordering);
  audit_fallback(corpus, fallback);

  print_counts("evaluation_snapshot", snapshots);
  print_counts("feature_bit_exactness", feature_bits);
  print_counts("scalar_evaluation", evaluations);
  std::cout << "maximum_feature_delta=" << std::setprecision(9)
            << maximum_feature_delta << '\n';
  print_counts("move_order", ordering);
  print_counts("fallback", fallback);

  const std::array<std::uint64_t, 5> node_budgets{{1, 16, 64, 256, 1024}};
  const std::size_t fixed_node_count =
      std::min<std::size_t>(corpus.size(), 200);
  const std::vector<CorpusState> fixed_node_corpus(
      corpus.begin(), corpus.begin() + fixed_node_count);
  for (const std::uint64_t nodes : node_budgets) {
    AuditCounts actions;
    AuditCounts scores;
    AuditCounts stats;
    audit_fixed_nodes(fixed_node_corpus, nodes, actions, scores, stats);
    print_counts("fixed_nodes_" + std::to_string(nodes) + "_action", actions);
    print_counts("fixed_nodes_" + std::to_string(nodes) + "_score", scores);
    print_counts("fixed_nodes_" + std::to_string(nodes) + "_stats", stats);
  }

  const std::size_t depth_one_count = std::min<std::size_t>(corpus.size(), 64);
  const std::vector<CorpusState> depth_one_corpus(
      corpus.begin(), corpus.begin() + depth_one_count);
  AuditCounts depth_one_actions;
  AuditCounts depth_one_scores;
  AuditCounts depth_one_stats;
  AuditCounts depth_one_incomplete;
  audit_fixed_depth(depth_one_corpus, 1, depth_one_actions, depth_one_scores,
                    depth_one_stats, depth_one_incomplete);
  print_counts("fixed_depth_1_action", depth_one_actions);
  print_counts("fixed_depth_1_score", depth_one_scores);
  print_counts("fixed_depth_1_stats", depth_one_stats);
  print_counts("fixed_depth_1_incomplete", depth_one_incomplete);

  const std::size_t depth_two_count = std::min<std::size_t>(corpus.size(), 24);
  const std::vector<CorpusState> depth_two_corpus(
      corpus.begin(), corpus.begin() + depth_two_count);
  AuditCounts depth_two_actions;
  AuditCounts depth_two_scores;
  AuditCounts depth_two_stats;
  AuditCounts depth_two_incomplete;
  audit_fixed_depth(depth_two_corpus, 2, depth_two_actions, depth_two_scores,
                    depth_two_stats, depth_two_incomplete);
  print_counts("fixed_depth_2_action", depth_two_actions);
  print_counts("fixed_depth_2_score", depth_two_scores);
  print_counts("fixed_depth_2_stats", depth_two_stats);
  print_counts("fixed_depth_2_incomplete", depth_two_incomplete);

  return 0;
}
