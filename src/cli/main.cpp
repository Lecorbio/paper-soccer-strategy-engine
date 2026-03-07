#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "papersoccer/debug.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

std::string player_to_string(ps::Player player) {
  return player == ps::Player::One ? "Player 1" : "Player 2";
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

bool parse_index(const std::string &input, std::size_t max, std::size_t &out_index) {
  try {
    std::size_t parsed_chars = 0;
    const auto value = std::stoul(input, &parsed_chars);
    if (parsed_chars != input.size()) {
      return false;
    }
    if (value >= max) {
      return false;
    }
    out_index = static_cast<std::size_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

}  // namespace

int main() {
  ps::GameState state = ps::make_initial_state();
  bool auto_print_board = true;

  std::cout << "Paper Soccer CLI (Kurnik-style baseline)\n";
  std::cout << "Type 'h' for help. Auto-print is ON.\n";

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
