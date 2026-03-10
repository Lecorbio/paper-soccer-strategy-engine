#include <cstdint>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/debug.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

enum class ControllerKind { Human, RandomBot };

struct CliConfig {
  ControllerKind player_one{ControllerKind::Human};
  ControllerKind player_two{ControllerKind::Human};
  std::uint64_t base_seed{ps::RandomBot::default_seed()};
};

std::string player_to_string(ps::Player player) {
  return player == ps::Player::One ? "Player 1" : "Player 2";
}

std::string controller_to_string(ControllerKind controller) {
  return controller == ControllerKind::Human ? "Human" : "RandomBot";
}

std::string format_position(ps::Point point) {
  return "(" + std::to_string(point.y) + ", " + std::to_string(point.x) + ")";
}

void print_help() {
  std::cout << "Commands:\n";
  std::cout << "  <index>  play move with that index\n";
  std::cout << "  b        print the current board\n";
  std::cout << "  a        toggle automatic board printing\n";
  std::cout << "  h        show this help\n";
  std::cout << "  q        quit\n";
}

bool parse_unsigned(const std::string &input, std::size_t max, std::size_t &out_value) {
  try {
    std::size_t parsed_chars = 0;
    const auto value = std::stoul(input, &parsed_chars);
    if (parsed_chars != input.size()) {
      return false;
    }
    if (value >= max) {
      return false;
    }
    out_value = static_cast<std::size_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_index(const std::string &input, std::size_t max, std::size_t &out_index) {
  return parse_unsigned(input, max, out_index);
}

bool parse_seed(const std::string &input, std::uint64_t &out_seed) {
  try {
    std::size_t parsed_chars = 0;
    const auto value = std::stoull(input, &parsed_chars);
    if (parsed_chars != input.size()) {
      return false;
    }
    out_seed = static_cast<std::uint64_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

std::string prompt_line(const std::string &prompt) {
  std::cout << prompt;
  std::string input;
  if (!std::getline(std::cin, input)) {
    throw std::runtime_error("input stream closed");
  }
  return input;
}

std::size_t prompt_choice(const std::string &prompt, std::size_t option_count) {
  while (true) {
    std::size_t choice = 0;
    const std::string input = prompt_line(prompt);
    if (parse_unsigned(input, option_count + 1, choice) && choice >= 1) {
      return choice;
    }
    std::cout << "Invalid selection. Enter a number between 1 and " << option_count << ".\n";
  }
}

std::uint64_t prompt_seed(std::uint64_t default_seed) {
  while (true) {
    const std::string input = prompt_line(
        "RandomBot base seed [" + std::to_string(default_seed) + "]: ");
    if (input.empty()) {
      return default_seed;
    }

    std::uint64_t seed = default_seed;
    if (parse_seed(input, seed)) {
      return seed;
    }
    std::cout << "Invalid seed. Enter a non-negative integer.\n";
  }
}

CliConfig prompt_config() {
  std::cout << "Select mode:\n";
  std::cout << "  [1] Human vs Human\n";
  std::cout << "  [2] Human vs RandomBot\n";
  std::cout << "  [3] RandomBot vs RandomBot\n";

  CliConfig config;
  const std::size_t mode = prompt_choice("Mode: ", 3);
  if (mode == 1) {
    return config;
  }

  config.base_seed = prompt_seed(ps::RandomBot::default_seed());
  if (mode == 2) {
    const std::size_t side = prompt_choice("Play as: [1] Player 1, [2] Player 2: ", 2);
    config.player_one = (side == 1) ? ControllerKind::Human : ControllerKind::RandomBot;
    config.player_two = (side == 1) ? ControllerKind::RandomBot : ControllerKind::Human;
    return config;
  }

  config.player_one = ControllerKind::RandomBot;
  config.player_two = ControllerKind::RandomBot;
  return config;
}

ControllerKind controller_for_player(const CliConfig &config, ps::Player player) {
  return player == ps::Player::One ? config.player_one : config.player_two;
}

std::uint64_t bot_seed(std::uint64_t base_seed, ps::Player player) {
  return player == ps::Player::One ? base_seed : base_seed + 1;
}

ps::Bot *bot_for_player(const CliConfig &config, ps::Player player,
                        std::unique_ptr<ps::Bot> &player_one_bot,
                        std::unique_ptr<ps::Bot> &player_two_bot) {
  if (controller_for_player(config, player) == ControllerKind::Human) {
    return nullptr;
  }
  return player == ps::Player::One ? player_one_bot.get() : player_two_bot.get();
}

}  // namespace

int main() {
  CliConfig config;
  try {
    config = prompt_config();
  } catch (const std::exception &error) {
    std::cout << "\n" << error.what() << ". Exiting.\n";
    return 0;
  }

  std::unique_ptr<ps::Bot> player_one_bot;
  std::unique_ptr<ps::Bot> player_two_bot;
  if (config.player_one == ControllerKind::RandomBot) {
    player_one_bot = std::make_unique<ps::RandomBot>(bot_seed(config.base_seed, ps::Player::One));
  }
  if (config.player_two == ControllerKind::RandomBot) {
    player_two_bot = std::make_unique<ps::RandomBot>(bot_seed(config.base_seed, ps::Player::Two));
  }

  ps::GameState state = ps::make_initial_state();
  bool auto_print_board = true;

  std::cout << "Paper Soccer CLI (Kurnik-style baseline)\n";
  std::cout << "Type 'h' for help. Auto-print is ON.\n";
  std::cout << "Player 1: " << controller_to_string(config.player_one) << "\n";
  std::cout << "Player 2: " << controller_to_string(config.player_two) << "\n";
  if (player_one_bot || player_two_bot) {
    std::cout << "RandomBot base seed: " << config.base_seed << "\n";
  }

  while (!ps::is_terminal(state)) {
    auto moves = ps::legal_moves(state);
    if (moves.empty()) {
      state.status =
          (state.to_move == ps::Player::One) ? ps::Status::WonByTwo : ps::Status::WonByOne;
      break;
    }

    if (auto_print_board) {
      std::cout << "\n" << ps::render_ascii(state);
    }

    if (ps::Bot *bot = bot_for_player(config, state.to_move, player_one_bot, player_two_bot)) {
      const ps::Move chosen_move = bot->choose_move(state);
      std::cout << "\n" << player_to_string(state.to_move) << " (" << bot->name()
                << ") chooses -> " << format_position(chosen_move.to) << "\n";
      state = ps::apply_move(state, chosen_move);
      continue;
    }

    std::cout << "\n" << player_to_string(state.to_move) << " to move.\n";
    std::cout << "Ball at " << format_position(state.ball) << " [row, column]\n";
    std::cout << "Legal moves:\n";
    for (std::size_t i = 0; i < moves.size(); ++i) {
      std::cout << "  [" << i << "] -> " << format_position(moves[i].to) << "\n";
    }

    std::cout << "Choose move index (b/a/h/q): ";
    std::string input;
    if (!std::getline(std::cin, input)) {
      std::cout << "\nInput stream closed. Exiting.\n";
      return 0;
    }

    if (input == "q" || input == "Q") {
      std::cout << "Quitting.\n";
      return 0;
    }
    if (input == "h" || input == "H") {
      print_help();
      continue;
    }
    if (input == "b" || input == "B") {
      std::cout << "\n" << ps::render_ascii(state);
      continue;
    }
    if (input == "a" || input == "A") {
      auto_print_board = !auto_print_board;
      std::cout << "Auto-print is now " << (auto_print_board ? "ON" : "OFF") << ".\n";
      continue;
    }

    std::size_t move_index = 0;
    if (!parse_index(input, moves.size(), move_index)) {
      std::cout << "Invalid input. Enter a valid move index, 'b', 'a', 'h', or 'q'.\n";
      continue;
    }

    try {
      state = ps::apply_move(state, moves[move_index]);
    } catch (const std::exception &error) {
      std::cout << "Move rejected: " << error.what() << "\n";
    }
  }

  std::cout << "\nFinal board:\n" << ps::render_ascii(state);

  const std::optional<ps::Player> winning_player = ps::winner(state);
  if (winning_player.has_value()) {
    std::cout << "\nWinner: " << player_to_string(*winning_player) << "\n";
  } else {
    std::cout << "\nGame ended with no winner.\n";
  }

  const std::size_t path_length = state.path.empty() ? 0 : state.path.size() - 1;
  std::cout << "Final path length: " << path_length << " moves\n";
  return 0;
}
