#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/match.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

struct ExportConfig {
  std::uint64_t base_seed{ps::RandomBot::default_seed()};
  std::size_t max_plies{512};
};

std::string player_to_json(ps::Player player) {
  return player == ps::Player::One ? "one" : "two";
}

std::string status_to_json(ps::Status status) {
  switch (status) {
    case ps::Status::InProgress:
      return "inProgress";
    case ps::Status::WonByOne:
      return "wonByOne";
    case ps::Status::WonByTwo:
      return "wonByTwo";
  }
  return "unknown";
}

std::uint64_t parse_uint64(const std::string &input, const char *field_name) {
  try {
    std::size_t parsed_chars = 0;
    const auto value = std::stoull(input, &parsed_chars);
    if (parsed_chars != input.size()) {
      throw std::invalid_argument("trailing input");
    }
    return static_cast<std::uint64_t>(value);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("invalid ") + field_name + ": " + input);
  }
}

std::size_t parse_size(const std::string &input, const char *field_name) {
  try {
    std::size_t parsed_chars = 0;
    const auto value = std::stoull(input, &parsed_chars);
    if (parsed_chars != input.size()) {
      throw std::invalid_argument("trailing input");
    }
    return static_cast<std::size_t>(value);
  } catch (const std::exception &) {
    throw std::invalid_argument(std::string("invalid ") + field_name + ": " + input);
  }
}

ExportConfig parse_args(int argc, char **argv) {
  ExportConfig config;
  if (argc > 3) {
    throw std::invalid_argument("usage: papersoccer_replay_export [base-seed] [max-plies]");
  }
  if (argc >= 2) {
    config.base_seed = parse_uint64(argv[1], "base seed");
  }
  if (argc >= 3) {
    config.max_plies = parse_size(argv[2], "max plies");
  }
  return config;
}

std::uint64_t bot_seed(std::uint64_t base_seed, ps::Player player) {
  return player == ps::Player::One ? base_seed : base_seed + 1;
}

void write_point(std::ostream &out, ps::Point point) {
  out << "{\"x\":" << point.x << ",\"y\":" << point.y << "}";
}

void write_replay_json(std::ostream &out, const ExportConfig &config,
                       const ps::Match &match, bool truncated) {
  const ps::GameState &state = match.state();
  const std::vector<ps::PlayedMove> &moves = match.history();
  out << "{\n";
  out << "  \"schema\": \"papersoccer.replay.v2\",\n";
  out << "  \"rules\": {\"width\": " << state.config.width
      << ", \"height\": " << state.config.height << "},\n";
  out << "  \"players\": {\n";
  out << "    \"one\": {\"kind\": \"RandomBot\", \"seed\": \""
      << bot_seed(config.base_seed, ps::Player::One) << "\"},\n";
  out << "    \"two\": {\"kind\": \"RandomBot\", \"seed\": \""
      << bot_seed(config.base_seed, ps::Player::Two) << "\"}\n";
  out << "  },\n";
  out << "  \"start\": ";
  write_point(out, ps::Point{state.config.width / 2, state.config.height / 2 + 1});
  out << ",\n";
  out << "  \"status\": \"" << status_to_json(state.status) << "\",\n";
  out << "  \"winner\": ";
  const std::optional<ps::Player> winning_player = ps::winner(state);
  if (winning_player.has_value()) {
    out << "\"" << player_to_json(*winning_player) << "\"";
  } else {
    out << "null";
  }
  out << ",\n";
  out << "  \"truncated\": " << (truncated ? "true" : "false") << ",\n";
  out << "  \"moves\": [\n";
  for (std::size_t i = 0; i < moves.size(); ++i) {
    const ps::PlayedMove &move = moves[i];
    out << "    {\"ply\": " << move.ply << ", \"player\": \""
        << player_to_json(move.player) << "\", \"from\": ";
    write_point(out, move.from);
    out << ", \"to\": ";
    write_point(out, move.to);
    out << ", \"extraTurn\": " << (move.extra_turn ? "true" : "false")
        << ", \"statusAfter\": \"" << status_to_json(move.status_after) << "\"}";
    if (i + 1 != moves.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
}

}  // namespace

int main(int argc, char **argv) {
  ExportConfig config;
  try {
    config = parse_args(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    return 2;
  }

  ps::RandomBot player_one_bot(bot_seed(config.base_seed, ps::Player::One));
  ps::RandomBot player_two_bot(bot_seed(config.base_seed, ps::Player::Two));
  ps::Match match;
  bool truncated = false;

  while (!ps::is_terminal(match.state())) {
    if (match.history().size() >= config.max_plies) {
      truncated = true;
      break;
    }

    const ps::Player player = match.state().to_move;
    ps::Bot &bot = player == ps::Player::One ? static_cast<ps::Bot &>(player_one_bot)
                                             : static_cast<ps::Bot &>(player_two_bot);
    (void)match.play(bot.choose_move(match.state()));
  }

  write_replay_json(std::cout, config, match, truncated);
  return 0;
}
