// Fresh procedural successor-boundary labels for the Rank-4/Jacek hybrid.
//
// This tool intentionally includes the frozen Rank-4 search as a teacher.  It
// does not read replay banks, arena payloads, or the campaign validation/final
// opening banks.  Every root is generated from the public rules and the seed
// supplied on the command line.
#define PAPER_SOCCER_TURN_ACTION_V2_NO_MAIN
#include "../submissions/codingame/bots/rank_4/bot.cpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace ps = papersoccer;
namespace teacher = papersoccer::turn_action_v2;

namespace {

constexpr std::uint64_t kDefaultSeed = 0x8f5a2e10b7c3496dULL;

struct SplitMix64 {
  std::uint64_t state{};

  std::uint64_t next() noexcept {
    std::uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  std::size_t index(std::size_t bound) {
    if (bound == 0) {
      throw std::invalid_argument("random bound must be positive");
    }
    const std::uint64_t threshold =
        static_cast<std::uint64_t>(-static_cast<std::uint64_t>(bound)) %
        static_cast<std::uint64_t>(bound);
    while (true) {
      const std::uint64_t value = next();
      if (value >= threshold) {
        return static_cast<std::size_t>(value % bound);
      }
    }
  }
};

struct Options {
  std::size_t roots{160};
  std::size_t max_actions{32};
  std::uint64_t low_nodes{30'000};
  std::uint64_t high_nodes{100'000};
  std::uint64_t seed{kDefaultSeed};
  std::string output;
  bool mirror{true};
};

struct Endpoint {
  ps::GameState state;
  std::string action;
};

std::uint64_t parse_u64(const std::string &text, const char *name) {
  std::size_t consumed = 0;
  const std::uint64_t value = std::stoull(text, &consumed, 0);
  if (consumed != text.size()) {
    throw std::invalid_argument(std::string("invalid ") + name);
  }
  return value;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("missing value after " + flag);
      }
      return argv[index];
    };
    if (flag == "--roots") {
      options.roots = static_cast<std::size_t>(parse_u64(value(), "roots"));
    } else if (flag == "--max-actions") {
      options.max_actions =
          static_cast<std::size_t>(parse_u64(value(), "max-actions"));
    } else if (flag == "--low-nodes") {
      options.low_nodes = parse_u64(value(), "low-nodes");
    } else if (flag == "--high-nodes") {
      options.high_nodes = parse_u64(value(), "high-nodes");
    } else if (flag == "--seed") {
      options.seed = parse_u64(value(), "seed");
    } else if (flag == "--output") {
      options.output = value();
    } else if (flag == "--mirror") {
      const std::string enabled = value();
      if (enabled != "0" && enabled != "1") {
        throw std::invalid_argument("--mirror must be 0 or 1");
      }
      options.mirror = enabled == "1";
    } else {
      throw std::invalid_argument("unknown option: " + flag);
    }
  }
  if (options.roots == 0 || options.max_actions < 2 ||
      options.low_nodes == 0 || options.high_nodes < options.low_nodes ||
      options.output.empty()) {
    throw std::invalid_argument(
        "usage: ranker_samples --output FILE [--roots N] "
        "[--max-actions N] [--low-nodes N] [--high-nodes N] "
        "[--seed N] [--mirror 0|1]");
  }
  return options;
}

ps::RulesConfig codingame_rules() {
  ps::RulesConfig rules;
  rules.goal_rule = ps::GoalRule::OwnGoalsAllowed;
  rules.blocked_rule = ps::BlockedRule::MoverLoses;
  return rules;
}

ps::GameState generate_boundary_root(std::uint64_t seed,
                                     std::size_t completed_turns) {
  SplitMix64 random{seed};
  for (std::size_t attempt = 0; attempt < 256; ++attempt) {
    ps::GameState state = ps::make_initial_state(codingame_rules());
    bool failed = false;
    for (std::size_t turn = 0; turn < completed_turns; ++turn) {
      const ps::Player mover = state.to_move;
      do {
        const std::vector<ps::Move> legal = ps::legal_moves(state);
        if (legal.empty()) {
          failed = true;
          break;
        }
        state = ps::apply_move(state, legal[random.index(legal.size())]);
      } while (!ps::is_terminal(state) && state.to_move == mover);
      if (failed || ps::is_terminal(state)) {
        failed = true;
        break;
      }
    }
    if (!failed && !ps::is_terminal(state) &&
        ps::legal_moves(state).size() >= 2) {
      return state;
    }
    random.state ^= 0xd1b54a32d192ed03ULL + attempt;
  }
  throw std::runtime_error("could not generate a nonterminal boundary root");
}

ps::Point mirror_point(ps::Point point, const ps::RulesConfig &rules) {
  point.x = rules.width - point.x;
  return point;
}

ps::GameState mirror_state(const ps::GameState &state) {
  ps::GameState result = state;
  result.ball = mirror_point(state.ball, state.config);
  result.path.clear();
  result.used_segments.clear();
  result.visit_count.clear();
  result.path.reserve(state.path.size());
  for (const ps::Point point : state.path) {
    const ps::Point mirrored = mirror_point(point, state.config);
    result.path.push_back(mirrored);
    ++result.visit_count[mirrored];
  }
  for (const ps::Segment &segment : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{mirror_point(segment.a, state.config),
                    mirror_point(segment.b, state.config)});
  }
  return result;
}

ps::Status swap_status(ps::Status status) {
  if (status == ps::Status::WonByOne) return ps::Status::WonByTwo;
  if (status == ps::Status::WonByTwo) return ps::Status::WonByOne;
  return status;
}

ps::Point rotate_point(ps::Point point, const ps::RulesConfig &rules) {
  return {rules.width - point.x, rules.height + 2 - point.y};
}

ps::GameState rotate_and_swap(const ps::GameState &state) {
  ps::GameState result = state;
  result.ball = rotate_point(state.ball, state.config);
  result.to_move = ps::opponent(state.to_move);
  result.status = swap_status(state.status);
  result.path.clear();
  result.used_segments.clear();
  result.visit_count.clear();
  result.path.reserve(state.path.size());
  for (const ps::Point point : state.path) {
    const ps::Point rotated = rotate_point(point, state.config);
    result.path.push_back(rotated);
    ++result.visit_count[rotated];
  }
  for (const ps::Segment &segment : state.used_segments) {
    result.used_segments.insert(
        ps::Segment{rotate_point(segment.a, state.config),
                    rotate_point(segment.b, state.config)});
  }
  return result;
}

std::string boundary_key(const ps::GameState &state) {
  std::vector<ps::Segment> segments(state.used_segments.begin(),
                                    state.used_segments.end());
  std::sort(segments.begin(), segments.end(),
            [](const ps::Segment &left, const ps::Segment &right) {
              return left.a < right.a ||
                     (!(right.a < left.a) && left.b < right.b);
            });
  std::ostringstream out;
  out << state.ball.x << ',' << state.ball.y << ','
      << (state.to_move == ps::Player::One ? '1' : '2') << ':';
  for (const ps::Segment &segment : segments) {
    out << segment.a.x << ',' << segment.a.y << '-' << segment.b.x << ','
        << segment.b.y << ';';
  }
  return out.str();
}

void enumerate_from_first(const ps::GameState &state, ps::Player mover,
                          std::string action, std::size_t edge_depth,
                          std::size_t limit, std::vector<Endpoint> &endpoints,
                          std::unordered_set<std::string> &seen) {
  if (endpoints.size() >= limit || edge_depth >= 48) {
    return;
  }
  for (const ps::Move move : ps::legal_moves(state)) {
    if (endpoints.size() >= limit) {
      return;
    }
    ps::GameState next = ps::apply_move(state, move);
    std::string next_action = action;
    next_action.push_back(teacher::encode_direction(state.ball, move.to));
    if (ps::is_terminal(next)) {
      // Goals and blocks are handled by exact tactical classes, not learned
      // ordering.  Excluding them prevents the ranker from displacing proof.
      continue;
    }
    if (next.to_move != mover) {
      const std::string key = boundary_key(next);
      if (seen.insert(key).second) {
        endpoints.push_back(Endpoint{std::move(next), std::move(next_action)});
      }
      continue;
    }
    enumerate_from_first(next, mover, std::move(next_action), edge_depth + 1,
                         limit, endpoints, seen);
  }
}

std::vector<Endpoint> enumerate_endpoints(const ps::GameState &root,
                                          std::size_t max_actions) {
  const std::vector<ps::Move> first_moves = ps::legal_moves(root);
  std::vector<Endpoint> result;
  std::unordered_set<std::string> seen;
  if (first_moves.empty()) {
    return result;
  }
  const std::size_t per_first =
      std::max<std::size_t>(1, (max_actions + first_moves.size() - 1) /
                                   first_moves.size());
  for (const ps::Move first : first_moves) {
    if (result.size() >= max_actions) {
      break;
    }
    ps::GameState next = ps::apply_move(root, first);
    std::string action(1, teacher::encode_direction(root.ball, first.to));
    const std::size_t branch_limit =
        std::min(max_actions, result.size() + per_first);
    if (ps::is_terminal(next)) {
      continue;
    }
    if (next.to_move != root.to_move) {
      const std::string key = boundary_key(next);
      if (seen.insert(key).second) {
        result.push_back(Endpoint{std::move(next), std::move(action)});
      }
    } else {
      enumerate_from_first(next, root.to_move, std::move(action), 1,
                           branch_limit, result, seen);
    }
  }
  return result;
}

struct TeacherResult {
  int score{};
  std::uint32_t depth{};
  std::uint64_t nodes{};
};

TeacherResult teacher_value(const ps::GameState &state,
                            std::uint64_t max_nodes) {
  teacher::SearchConfig config;
  config.max_nodes = max_nodes;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  teacher::CompleteTurnSearch search(state, config);
  (void)search.run();
  const teacher::SearchStats &stats = search.stats();
  return TeacherResult{stats.root_score, stats.completed_turn_depth,
                       stats.nodes};
}

teacher::EvaluationSnapshot features(const ps::GameState &state) {
  teacher::SearchConfig config;
  config.max_nodes = 1;
  config.replay_value_blend_percent = 15;
  config.teacher_residual_weight_percent = 100;
  teacher::CompleteTurnSearch search(state, config);
  return search.evaluation_snapshot();
}

std::string split_for(std::size_t root_index) {
  // Fixed 70/15/15 whole-root split.  Mirrored variants remain in the same
  // split because the split is assigned before symmetry expansion.  Divide
  // by the four-depth round-robin first so every split contains every depth.
  const std::size_t bucket = (root_index / 4) % 20;
  if (bucket < 14) {
    return "train";
  }
  if (bucket < 17) {
    return "validation";
  }
  return "test";
}

void write_candidate(std::ostream &out, const std::string &root_id,
                     const std::string &split, const std::string &variant,
                     std::size_t root_depth, std::size_t candidate,
                     const Endpoint &endpoint,
                     std::uint64_t low_nodes, std::uint64_t high_nodes) {
  const teacher::EvaluationSnapshot snapshot = features(endpoint.state);
  const TeacherResult low = teacher_value(endpoint.state, low_nodes);
  const TeacherResult high = teacher_value(endpoint.state, high_nodes);
  const int sign = endpoint.state.to_move == ps::Player::One ? 1 : -1;
  out << "{\"schema\":\"papersoccer.rank4-jacek-successor.v2\","
      << "\"root_id\":\"" << root_id << "\",\"split\":\"" << split
      << "\",\"variant\":\"" << variant << "\",\"root_depth\":"
      << root_depth << ",\"candidate\":"
      << candidate << ",\"action\":\"" << endpoint.action
      << "\",\"features\":[";
  out << std::setprecision(9);
  for (std::size_t index = 0; index < snapshot.features.size(); ++index) {
    if (index != 0) {
      out << ',';
    }
    out << snapshot.features[index];
  }
  out << "],\"anchor_score\":" << snapshot.anchor_score
      << ",\"root_mover_sign\":" << -sign
      << ",\"successor_mover_sign\":" << sign
      << ",\"low_score\":" << sign * low.score
      << ",\"high_score\":" << sign * high.score
      << ",\"low_depth\":" << low.depth << ",\"high_depth\":"
      << high.depth << ",\"low_nodes_used\":" << low.nodes
      << ",\"high_nodes_used\":" << high.nodes << "}\n";
}

void write_variant(std::ostream &out, const ps::GameState &root,
                   const std::string &root_id, const std::string &split,
                   const std::string &variant, std::size_t root_depth,
                   const Options &options) {
  const std::vector<Endpoint> endpoints =
      enumerate_endpoints(root, options.max_actions);
  for (std::size_t candidate = 0; candidate < endpoints.size(); ++candidate) {
    write_candidate(out, root_id, split, variant, root_depth, candidate,
                    endpoints[candidate], options.low_nodes,
                    options.high_nodes);
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::ofstream output(options.output, std::ios::binary);
    if (!output) {
      throw std::runtime_error("could not open output: " + options.output);
    }
    output << "{\"schema\":\"papersoccer.rank4-jacek-corpus-meta.v2\","
           << "\"seed\":" << options.seed << ",\"roots\":"
           << options.roots << ",\"max_actions\":" << options.max_actions
           << ",\"low_nodes\":" << options.low_nodes
           << ",\"high_nodes\":" << options.high_nodes
           << ",\"mirror\":" << (options.mirror ? "true" : "false")
           << ",\"color_swap\":true,\"variants\":[\"base\",\"mirror\","
              "\"rotate\",\"rotate_mirror\"]"
           << ",\"rules\":\"8x10;own-goals-allowed;mover-loses\","
           << "\"teacher\":\"frozen-rank-4-proof-off\"}\n";

    SplitMix64 seeds{options.seed};
    constexpr std::array<std::size_t, 4> kTurnDepths{{4, 8, 12, 20}};
    for (std::size_t root_index = 0; root_index < options.roots;
         ++root_index) {
      const std::uint64_t root_seed = seeds.next();
      const std::size_t depth = kTurnDepths[root_index % kTurnDepths.size()];
      const ps::GameState root = generate_boundary_root(root_seed, depth);
      std::ostringstream identity;
      identity << "fresh-r" << std::setw(4) << std::setfill('0') << root_index
               << "-d" << std::setw(2) << depth << '-' << std::hex
               << std::setw(16) << root_seed;
      const std::string root_id = identity.str();
      const std::string split = split_for(root_index);
      write_variant(output, root, root_id, split, "base", depth, options);
      if (options.mirror) {
        write_variant(output, mirror_state(root), root_id, split, "mirror",
                      depth, options);
      }
      const ps::GameState rotated = rotate_and_swap(root);
      write_variant(output, rotated, root_id, split, "rotate", depth,
                    options);
      if (options.mirror) {
        write_variant(output, mirror_state(rotated), root_id, split,
                      "rotate_mirror", depth, options);
      }
      std::cerr << "ranker-corpus root " << (root_index + 1) << '/'
                << options.roots << '\n';
    }
  } catch (const std::exception &error) {
    std::cerr << "ranker-corpus error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
