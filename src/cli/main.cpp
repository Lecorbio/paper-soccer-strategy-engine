#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "papersoccer/bot.hpp"
#include "papersoccer/debug.hpp"
#include "papersoccer/rules.hpp"

namespace ps = papersoccer;

namespace {

enum class ControllerKind {
  Human,
  RandomBot,
  MctsBot,
  AlphaBetaBot,
  JacekInspiredBot,
  JacekReplayBfmBot,
};

struct CliConfig {
  ps::RulesConfig rules{};
  ControllerKind player_one{ControllerKind::Human};
  ControllerKind player_two{ControllerKind::Human};
  std::uint64_t base_seed{ps::RandomBot::default_seed()};
  std::uint64_t mcts_base_seed{ps::RandomBot::default_seed()};
  std::uint32_t mcts_iterations{2000};
  std::uint32_t alpha_beta_depth{6};
  std::uint64_t alpha_beta_max_nodes{100'000};
  std::uint32_t jacek_depth{6};
  std::uint64_t jacek_max_nodes{20'000};
  ps::JacekReplayBfmConfig jacek_replay_bfm{};
};

std::string player_to_string(ps::Player player) {
  return player == ps::Player::One ? "Player 1" : "Player 2";
}

std::string controller_to_string(ControllerKind controller) {
  switch (controller) {
    case ControllerKind::Human:
      return "Human";
    case ControllerKind::RandomBot:
      return "RandomBot";
    case ControllerKind::MctsBot:
      return "MctsBot";
    case ControllerKind::AlphaBetaBot:
      return "AlphaBetaBot";
    case ControllerKind::JacekInspiredBot:
      return "JacekInspiredBot";
    case ControllerKind::JacekReplayBfmBot:
      return "JacekReplayBfmBot";
  }
  return "Unknown";
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

bool parse_iterations(const std::string &input, std::uint32_t &out_iterations) {
  try {
    if (input.empty() || input.front() == '-') {
      return false;
    }
    std::size_t parsed_chars = 0;
    const auto value = std::stoull(input, &parsed_chars);
    if (parsed_chars != input.size() || value == 0 ||
        value > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    out_iterations = static_cast<std::uint32_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_alpha_beta_depth(const std::string &input,
                            std::uint32_t &out_depth) {
  std::uint32_t depth = 0;
  if (!parse_iterations(input, depth) ||
      depth > ps::AlphaBetaConfig::maximum_turn_depth) {
    return false;
  }
  out_depth = depth;
  return true;
}

bool parse_node_budget(const std::string &input,
                       std::uint64_t &out_node_budget) {
  try {
    if (input.empty() || input.front() == '-') {
      return false;
    }
    std::size_t parsed_chars = 0;
    const auto value = std::stoull(input, &parsed_chars);
    if (parsed_chars != input.size() || value == 0) {
      return false;
    }
    out_node_budget = static_cast<std::uint64_t>(value);
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

std::uint64_t prompt_seed(const std::string &label, std::uint64_t default_seed) {
  while (true) {
    const std::string input =
        prompt_line(label + " [" + std::to_string(default_seed) + "]: ");
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

std::uint32_t prompt_iterations(std::uint32_t default_iterations) {
  while (true) {
    const std::string input = prompt_line(
        "MctsBot iterations per move [" + std::to_string(default_iterations) + "]: ");
    if (input.empty()) {
      return default_iterations;
    }

    std::uint32_t iterations = default_iterations;
    if (parse_iterations(input, iterations)) {
      return iterations;
    }
    std::cout << "Invalid iteration count. Enter an integer greater than zero.\n";
  }
}

std::uint32_t prompt_search_depth(const std::string &bot_name,
                                  std::uint32_t default_depth) {
  while (true) {
    const std::string input = prompt_line(
        bot_name + " turn depth [" + std::to_string(default_depth) + "]: ");
    if (input.empty()) {
      return default_depth;
    }

    std::uint32_t depth = default_depth;
    if (parse_alpha_beta_depth(input, depth)) {
      return depth;
    }
    std::cout << "Invalid depth. Enter an integer between 1 and "
              << ps::AlphaBetaConfig::maximum_turn_depth << ".\n";
  }
}

std::uint64_t prompt_search_max_nodes(const std::string &bot_name,
                                      std::uint64_t default_max_nodes) {
  while (true) {
    const std::string input = prompt_line(
        bot_name + " node budget per move [" +
        std::to_string(default_max_nodes) + "]: ");
    if (input.empty()) {
      return default_max_nodes;
    }

    std::uint64_t max_nodes = default_max_nodes;
    if (parse_node_budget(input, max_nodes)) {
      return max_nodes;
    }
    std::cout << "Invalid node budget. Enter an integer greater than zero.\n";
  }
}

std::string prompt_required_path(const std::string &label) {
  while (true) {
    const std::string path = prompt_line(label + ": ");
    if (!path.empty()) {
      return path;
    }
    std::cout << "A non-empty model path is required.\n";
  }
}

std::uint32_t prompt_time_ms(std::uint32_t default_time_ms) {
  while (true) {
    const std::string input =
        prompt_line("JacekReplayBfmBot time budget in milliseconds [" +
                    std::to_string(default_time_ms) + "]: ");
    if (input.empty()) {
      return default_time_ms;
    }
    std::uint32_t parsed = default_time_ms;
    if (parse_iterations(input, parsed)) {
      return parsed;
    }
    std::cout << "Invalid time budget. Enter an integer greater than zero.\n";
  }
}

std::size_t prompt_bfm_tree_nodes(std::size_t default_max_nodes) {
  while (true) {
    const std::string input =
        prompt_line("JacekReplayBfmBot tree node budget [" +
                    std::to_string(default_max_nodes) + "]: ");
    if (input.empty()) {
      return default_max_nodes;
    }
    std::uint64_t parsed = 0;
    if (parse_node_budget(input, parsed) && parsed >= 2 &&
        parsed <= 1'000'000) {
      return static_cast<std::size_t>(parsed);
    }
    std::cout << "Invalid node budget. Enter an integer from 2 to 1000000.\n";
  }
}

ps::RulesConfig prompt_rules() {
  std::cout << "Select rules:\n";
  std::cout << "  [1] Normal demo (opponent goal only; player to move loses)\n";
  std::cout << "  [2] CodinGame (own goals allowed; mover loses)\n";
  const std::size_t choice = prompt_choice("Rules: ", 2);
  if (choice == 2) {
    return ps::RulesConfig{8, 10, ps::GoalRule::OwnGoalsAllowed,
                           ps::BlockedRule::MoverLoses};
  }
  return ps::RulesConfig{};
}

bool uses_codingame_rules(const ps::RulesConfig &rules) noexcept {
  return rules.width == 8 && rules.height == 10 &&
         rules.goal_rule == ps::GoalRule::OwnGoalsAllowed &&
         rules.blocked_rule == ps::BlockedRule::MoverLoses;
}

bool uses_controller(const CliConfig &config, ControllerKind controller) {
  return config.player_one == controller || config.player_two == controller;
}

CliConfig prompt_config() {
  std::cout << "Select mode:\n";
  std::cout << "  [1] Human vs Human\n";
  std::cout << "  [2] Human vs RandomBot\n";
  std::cout << "  [3] RandomBot vs RandomBot\n";
  std::cout << "  [4] Human vs MctsBot\n";
  std::cout << "  [5] RandomBot vs MctsBot\n";
  std::cout << "  [6] MctsBot vs MctsBot\n";
  std::cout << "  [7] Human vs AlphaBetaBot\n";
  std::cout << "  [8] RandomBot vs AlphaBetaBot\n";
  std::cout << "  [9] MctsBot vs AlphaBetaBot\n";
  std::cout << "  [10] AlphaBetaBot vs AlphaBetaBot\n";
  std::cout << "  [11] Human vs JacekInspiredBot\n";
  std::cout << "  [12] RandomBot vs JacekInspiredBot\n";
  std::cout << "  [13] MctsBot vs JacekInspiredBot\n";
  std::cout << "  [14] AlphaBetaBot vs JacekInspiredBot\n";
  std::cout << "  [15] JacekInspiredBot vs JacekInspiredBot\n";
  std::cout << "  [16] Human vs JacekReplayBfmBot\n";
  std::cout << "  [17] RandomBot vs JacekReplayBfmBot\n";
  std::cout << "  [18] MctsBot vs JacekReplayBfmBot\n";
  std::cout << "  [19] AlphaBetaBot vs JacekReplayBfmBot\n";
  std::cout << "  [20] JacekReplayBfmBot vs JacekReplayBfmBot\n";

  CliConfig config;
  const std::size_t mode = prompt_choice("Mode: ", 20);
  switch (mode) {
    case 1:
      break;
    case 2: {
      const std::size_t side =
          prompt_choice("Play as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::Human : ControllerKind::RandomBot;
      config.player_two =
          (side == 1) ? ControllerKind::RandomBot : ControllerKind::Human;
      break;
    }
    case 3:
      config.player_one = ControllerKind::RandomBot;
      config.player_two = ControllerKind::RandomBot;
      break;
    case 4: {
      const std::size_t side =
          prompt_choice("Play as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::Human : ControllerKind::MctsBot;
      config.player_two =
          (side == 1) ? ControllerKind::MctsBot : ControllerKind::Human;
      break;
    }
    case 5: {
      const std::size_t side =
          prompt_choice("MctsBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::MctsBot : ControllerKind::RandomBot;
      config.player_two =
          (side == 1) ? ControllerKind::RandomBot : ControllerKind::MctsBot;
      break;
    }
    case 6:
      config.player_one = ControllerKind::MctsBot;
      config.player_two = ControllerKind::MctsBot;
      break;
    case 7: {
      const std::size_t side =
          prompt_choice("Play as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::Human : ControllerKind::AlphaBetaBot;
      config.player_two =
          (side == 1) ? ControllerKind::AlphaBetaBot : ControllerKind::Human;
      break;
    }
    case 8: {
      const std::size_t side = prompt_choice(
          "AlphaBetaBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one = (side == 1) ? ControllerKind::AlphaBetaBot
                                      : ControllerKind::RandomBot;
      config.player_two = (side == 1) ? ControllerKind::RandomBot
                                      : ControllerKind::AlphaBetaBot;
      break;
    }
    case 9: {
      const std::size_t side = prompt_choice(
          "AlphaBetaBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one = (side == 1) ? ControllerKind::AlphaBetaBot
                                      : ControllerKind::MctsBot;
      config.player_two = (side == 1) ? ControllerKind::MctsBot
                                      : ControllerKind::AlphaBetaBot;
      break;
    }
    case 10:
      config.player_one = ControllerKind::AlphaBetaBot;
      config.player_two = ControllerKind::AlphaBetaBot;
      break;
    case 11: {
      const std::size_t side =
          prompt_choice("Play as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::Human
                      : ControllerKind::JacekInspiredBot;
      config.player_two =
          (side == 1) ? ControllerKind::JacekInspiredBot
                      : ControllerKind::Human;
      break;
    }
    case 12: {
      const std::size_t side = prompt_choice(
          "JacekInspiredBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::JacekInspiredBot
                      : ControllerKind::RandomBot;
      config.player_two =
          (side == 1) ? ControllerKind::RandomBot
                      : ControllerKind::JacekInspiredBot;
      break;
    }
    case 13: {
      const std::size_t side = prompt_choice(
          "JacekInspiredBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::JacekInspiredBot
                      : ControllerKind::MctsBot;
      config.player_two =
          (side == 1) ? ControllerKind::MctsBot
                      : ControllerKind::JacekInspiredBot;
      break;
    }
    case 14: {
      const std::size_t side = prompt_choice(
          "JacekInspiredBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          (side == 1) ? ControllerKind::JacekInspiredBot
                      : ControllerKind::AlphaBetaBot;
      config.player_two =
          (side == 1) ? ControllerKind::AlphaBetaBot
                      : ControllerKind::JacekInspiredBot;
      break;
    }
    case 15:
      config.player_one = ControllerKind::JacekInspiredBot;
      config.player_two = ControllerKind::JacekInspiredBot;
      break;
    case 16: {
      const std::size_t side =
          prompt_choice("Play as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one =
          side == 1 ? ControllerKind::Human : ControllerKind::JacekReplayBfmBot;
      config.player_two =
          side == 1 ? ControllerKind::JacekReplayBfmBot : ControllerKind::Human;
      break;
    }
    case 17: {
      const std::size_t side = prompt_choice(
          "JacekReplayBfmBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one = side == 1 ? ControllerKind::JacekReplayBfmBot
                                    : ControllerKind::RandomBot;
      config.player_two = side == 1 ? ControllerKind::RandomBot
                                    : ControllerKind::JacekReplayBfmBot;
      break;
    }
    case 18: {
      const std::size_t side = prompt_choice(
          "JacekReplayBfmBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one = side == 1 ? ControllerKind::JacekReplayBfmBot
                                    : ControllerKind::MctsBot;
      config.player_two = side == 1 ? ControllerKind::MctsBot
                                    : ControllerKind::JacekReplayBfmBot;
      break;
    }
    case 19: {
      const std::size_t side = prompt_choice(
          "JacekReplayBfmBot plays as: [1] Player 1, [2] Player 2: ", 2);
      config.player_one = side == 1 ? ControllerKind::JacekReplayBfmBot
                                    : ControllerKind::AlphaBetaBot;
      config.player_two = side == 1 ? ControllerKind::AlphaBetaBot
                                    : ControllerKind::JacekReplayBfmBot;
      break;
    }
    case 20:
      config.player_one = ControllerKind::JacekReplayBfmBot;
      config.player_two = ControllerKind::JacekReplayBfmBot;
      break;
    default:
      throw std::logic_error("unsupported CLI mode");
  }

  config.rules = prompt_rules();
  if (uses_controller(config, ControllerKind::JacekInspiredBot) &&
      uses_codingame_rules(config.rules)) {
    throw std::invalid_argument(
        "JacekInspiredBot requires the normal demo rules profile");
  }
  if (uses_controller(config, ControllerKind::JacekReplayBfmBot) &&
      !uses_codingame_rules(config.rules)) {
    throw std::invalid_argument(
        "JacekReplayBfmBot requires the CodinGame rules profile");
  }

  if (uses_controller(config, ControllerKind::RandomBot)) {
    config.base_seed =
        prompt_seed("RandomBot base seed", ps::RandomBot::default_seed());
  }
  if (uses_controller(config, ControllerKind::MctsBot)) {
    config.mcts_base_seed =
        prompt_seed("MctsBot base seed", ps::RandomBot::default_seed());
    config.mcts_iterations = prompt_iterations(2000);
  }
  if (uses_controller(config, ControllerKind::AlphaBetaBot)) {
    config.alpha_beta_depth = prompt_search_depth("AlphaBetaBot", 6);
    config.alpha_beta_max_nodes =
        prompt_search_max_nodes("AlphaBetaBot", 100'000);
  }
  if (uses_controller(config, ControllerKind::JacekInspiredBot)) {
    config.jacek_depth = prompt_search_depth("JacekInspiredBot", 6);
    config.jacek_max_nodes =
        prompt_search_max_nodes("JacekInspiredBot", 20'000);
  }
  if (uses_controller(config, ControllerKind::JacekReplayBfmBot)) {
    config.jacek_replay_bfm.model_path =
        prompt_required_path("JacekReplayBfmBot checkpoint path");
    config.jacek_replay_bfm.seed = prompt_seed("JacekReplayBfmBot base seed",
                                               ps::RandomBot::default_seed());
    config.jacek_replay_bfm.max_time_ms =
        prompt_time_ms(config.jacek_replay_bfm.max_time_ms);
    config.jacek_replay_bfm.max_tree_nodes =
        prompt_bfm_tree_nodes(config.jacek_replay_bfm.max_tree_nodes);
  }
  return config;
}

ControllerKind controller_for_player(const CliConfig &config, ps::Player player) {
  return player == ps::Player::One ? config.player_one : config.player_two;
}

std::uint64_t bot_seed(std::uint64_t base_seed, ps::Player player) {
  return player == ps::Player::One ? base_seed : base_seed + 1;
}

std::unique_ptr<ps::Bot> make_player_bot(const CliConfig &config,
                                         ps::Player player) {
  const ControllerKind controller = controller_for_player(config, player);
  if (controller == ControllerKind::Human) {
    return nullptr;
  }

  ps::BotConfig bot_config;
  switch (controller) {
    case ControllerKind::Human:
      throw std::logic_error("human controller cannot create a bot");
    case ControllerKind::RandomBot:
      bot_config.kind = ps::BotKind::Random;
      bot_config.seed = bot_seed(config.base_seed, player);
      break;
    case ControllerKind::MctsBot:
      bot_config.kind = ps::BotKind::Mcts;
      bot_config.seed = bot_seed(config.mcts_base_seed, player);
      bot_config.mcts_iterations = config.mcts_iterations;
      break;
    case ControllerKind::AlphaBetaBot:
      bot_config.kind = ps::BotKind::AlphaBeta;
      bot_config.alpha_beta_depth = config.alpha_beta_depth;
      bot_config.alpha_beta_max_nodes = config.alpha_beta_max_nodes;
      break;
    case ControllerKind::JacekInspiredBot:
      bot_config.kind = ps::BotKind::JacekInspired;
      bot_config.alpha_beta_depth = config.jacek_depth;
      bot_config.alpha_beta_max_nodes = config.jacek_max_nodes;
      break;
    case ControllerKind::JacekReplayBfmBot: {
      ps::JacekReplayBfmConfig bfm = config.jacek_replay_bfm;
      bfm.seed = bot_seed(config.jacek_replay_bfm.seed, player);
      return std::make_unique<ps::JacekReplayBfmBot>(std::move(bfm));
    }
  }
  return ps::make_bot(bot_config);
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
    return std::string_view(error.what()) == "input stream closed" ? 0 : 2;
  }

  std::unique_ptr<ps::Bot> player_one_bot;
  std::unique_ptr<ps::Bot> player_two_bot;
  try {
    player_one_bot = make_player_bot(config, ps::Player::One);
    player_two_bot = make_player_bot(config, ps::Player::Two);
  } catch (const std::exception &error) {
    std::cout << "\nCould not create controller: " << error.what()
              << ". Exiting.\n";
    return 2;
  }

  ps::GameState state = ps::make_initial_state(config.rules);
  bool auto_print_board = true;

  std::cout << "Paper Soccer CLI\n";
  std::cout << "Type 'h' for help. Auto-print is ON.\n";
  std::cout << "Player 1: " << controller_to_string(config.player_one) << "\n";
  std::cout << "Player 2: " << controller_to_string(config.player_two) << "\n";
  std::cout << "Rules: "
            << (uses_codingame_rules(config.rules) ? "CodinGame" : "Normal")
            << "\n";
  if (config.player_one == ControllerKind::RandomBot ||
      config.player_two == ControllerKind::RandomBot) {
    std::cout << "RandomBot base seed: " << config.base_seed << "\n";
  }
  if (config.player_one == ControllerKind::MctsBot ||
      config.player_two == ControllerKind::MctsBot) {
    std::cout << "MctsBot base seed: " << config.mcts_base_seed << "\n";
    std::cout << "MctsBot iterations per move: " << config.mcts_iterations << "\n";
  }
  if (config.player_one == ControllerKind::AlphaBetaBot ||
      config.player_two == ControllerKind::AlphaBetaBot) {
    std::cout << "AlphaBetaBot turn depth: " << config.alpha_beta_depth << "\n";
    std::cout << "AlphaBetaBot node budget per move: "
              << config.alpha_beta_max_nodes << "\n";
  }
  if (config.player_one == ControllerKind::JacekInspiredBot ||
      config.player_two == ControllerKind::JacekInspiredBot) {
    std::cout << "JacekInspiredBot turn depth: " << config.jacek_depth << "\n";
    std::cout << "JacekInspiredBot node budget per move: "
              << config.jacek_max_nodes << "\n";
    std::cout << "JacekInspiredBot model SHA-256: "
              << ps::JacekInspiredBot::model_sha256() << "\n";
  }
  if (config.player_one == ControllerKind::JacekReplayBfmBot ||
      config.player_two == ControllerKind::JacekReplayBfmBot) {
    const ps::JacekReplayBfmBot *bfm = nullptr;
    if (player_one_bot) {
      bfm = dynamic_cast<const ps::JacekReplayBfmBot *>(player_one_bot.get());
    }
    if (bfm == nullptr && player_two_bot) {
      bfm = dynamic_cast<const ps::JacekReplayBfmBot *>(player_two_bot.get());
    }
    std::cout << "JacekReplayBfmBot checkpoint: "
              << config.jacek_replay_bfm.model_path << "\n";
    std::cout << "JacekReplayBfmBot time budget: "
              << config.jacek_replay_bfm.max_time_ms << " ms\n";
    std::cout << "JacekReplayBfmBot tree node budget: "
              << config.jacek_replay_bfm.max_tree_nodes << "\n";
    if (bfm != nullptr) {
      std::cout << "JacekReplayBfmBot model SHA-256: " << bfm->model_sha256()
                << "\n";
    }
    std::cout << "JacekReplayBfmBot feature schema SHA-256: "
              << ps::JacekReplayBfmBot::feature_schema_sha256() << "\n";
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
      if (const auto *mcts_bot = dynamic_cast<const ps::MctsBot *>(bot)) {
        const ps::SearchStats stats = mcts_bot->last_search_stats();
        std::cout << "MCTS stats: iterations=" << stats.iterations
                  << ", nodes=" << stats.nodes
                  << ", simulated plies=" << stats.simulated_plies
                  << ", root visits=" << stats.total_root_visits
                  << ", reused visits=" << stats.reused_visits
                  << ", max depth=" << stats.max_depth
                  << ", proven nodes=" << stats.proven_nodes
                  << ", tactical probes=" << stats.tactical_probes
                  << ", tactical nodes=" << stats.tactical_nodes
                  << ", tactical solved positions="
                  << stats.tactical_solved_positions
                  << ", tactical depth cutoffs="
                  << stats.tactical_depth_cutoffs
                  << ", tactical node cutoffs="
                  << stats.tactical_node_cutoffs
                  << ", max tactical depth=" << stats.max_tactical_depth
                  << ", rebuilds=" << stats.rebuild_count
                  << ", estimated root value=" << stats.root_value;
        if (stats.proven_winner.has_value()) {
          std::cout << ", proven winner="
                    << player_to_string(*stats.proven_winner);
        }
        if (stats.expansion_saturated) {
          std::cout << ", tree saturated";
        }
        std::cout << "\n";
      } else if (const auto *bfm =
                     dynamic_cast<const ps::JacekReplayBfmBot *>(bot)) {
        const ps::JacekReplayBfmSearchStats &stats =
            bfm->last_search_stats();
        std::cout << "Jacek replay BFM stats: expansions=" << stats.expansions
                  << ", generated actions=" << stats.generated_actions
                  << ", retained actions=" << stats.retained_actions
                  << ", neural evaluations=" << stats.neural_evaluations
                  << ", visits=" << stats.visits
                  << ", tree nodes=" << stats.tree_nodes
                  << ", complete-turn depth=" << stats.max_complete_turn_depth
                  << ", tactical proofs=" << stats.tactical_proofs
                  << ", tactical solutions=" << stats.tactical_solutions
                  << ", truncations=" << stats.truncations
                  << ", root value=" << stats.root_value;
        if (stats.root_solved) {
          std::cout << ", root solved";
          if (stats.proven_winner.has_value()) {
            std::cout << ", proven winner="
                      << player_to_string(*stats.proven_winner);
          }
        }
        if (stats.deadline_reached) {
          std::cout << ", deadline reached";
        }
        if (stats.tree_cap_reached) {
          std::cout << ", tree cap reached";
        }
        if (stats.cached_continuation) {
          std::cout << ", cached continuation edge " << stats.current_edge_index
                    << "/" << stats.planned_action_length << " ("
                    << stats.cached_moves_remaining << " remaining)";
        }
        std::cout << "\n";
      } else {
        const ps::AlphaBetaSearchStats *stats = nullptr;
        std::string search_name;
        if (const auto *alpha_beta_bot =
                dynamic_cast<const ps::AlphaBetaBot *>(bot)) {
          stats = &alpha_beta_bot->last_search_stats();
          search_name = "Alpha-beta";
        } else if (const auto *jacek_bot =
                       dynamic_cast<const ps::JacekInspiredBot *>(bot)) {
          stats = &jacek_bot->last_search_stats();
          search_name = "Jacek-inspired neural alpha-beta";
        }
        if (stats == nullptr) {
          state = ps::apply_move(state, chosen_move);
          continue;
        }
        std::cout << search_name << " stats: depth="
                  << stats->completed_turn_depth << "/"
                  << stats->attempted_turn_depth << ", nodes=" << stats->nodes
                  << ", evaluations=" << stats->leaf_evaluations
                  << ", terminal nodes=" << stats->terminal_nodes
                  << ", cutoffs=" << stats->cutoffs << ", TT hits="
                  << stats->transposition_hits << "/"
                  << stats->transposition_probes
                  << ", max physical ply=" << stats->max_physical_ply
                  << ", physical-ply cutoffs="
                  << stats->physical_ply_cutoffs;
        if (stats->completed_turn_depth > 0) {
          std::cout << ", Player 1 root score=" << stats->root_score;
        } else {
          std::cout << ", Player 1 root score=unavailable";
        }
        std::cout << ", PV plies=" << stats->principal_variation.size();
        if (stats->budget_exhausted) {
          std::cout << ", node budget reached";
        }
        std::cout << "\n";
      }
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
