#define PAPER_SOCCER_JACEK_NATIVE_LATE_PACING_GAME_GATE_NO_MAIN
#include "../../tools/jacek_native_late_pacing_game_gate.cpp"

#include <cstdlib>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

struct RenderedRoot {
  std::string run;
  std::string population;
  std::string role;
  std::string game_id;
  int color{};
  std::size_t depth{};
  std::string state_id;
  std::string canonical_key;
  std::string transcript;
};

std::string make_valid_plan(bool population_imbalance = false) {
  std::vector<RenderedRoot> rows;
  std::set<std::string> canonical_keys;
  std::uint64_t attempt = 1U;
  std::size_t game_id = 10'000U;
  constexpr std::array<std::array<std::size_t, 2>, 4> kGroupCounts{{
      {{5U, 7U}}, {{4U, 7U}}, {{4U, 1U}}, {{3U, 1U}},
  }};
  for (std::size_t run = 0; run < 4U; ++run) {
    for (const std::string population : {"control", "trap"}) {
      for (int color = 0; color < 2; ++color) {
        std::size_t collected = 0U;
        std::size_t required =
            kGroupCounts[run][static_cast<std::size_t>(color)];
        if (population_imbalance && run == 0U && color == 0) {
          if (population == "control") {
            ++required;
          } else {
            --required;
          }
        }
        while (collected < required) {
          const std::size_t depth = 1U + static_cast<std::size_t>(attempt % 12U);
          const Opening opening = make_opening(
              restart_rules(), 0x123456789abcdef0ULL + attempt, depth);
          ++attempt;
          if ((opening.state.to_move == ps::Player::One ? 0 : 1) != color) {
            continue;
          }
          const auto active = active_features(opening.state);
          const auto reflected = active_features(reflected_state(opening.state));
          const std::string canonical =
              feature_id(reflected < active ? reflected : active);
          if (!canonical_keys.insert(canonical).second) continue;
          rows.push_back(RenderedRoot{
              "run-" + std::to_string(run), population,
              population == "control" ? "matched-winning-control"
                                      : "one-own-before-mate",
              std::to_string(game_id++), color, depth, feature_id(active),
              canonical, opening.transcript,
          });
          ++collected;
        }
      }
    }
  }
  std::sort(rows.begin(), rows.end(), [](const auto &left, const auto &right) {
    return std::tie(left.run, left.population, left.color, left.canonical_key,
                    left.game_id, left.depth) <
           std::tie(right.run, right.population, right.color,
                    right.canonical_key, right.game_id, right.depth);
  });
  std::ostringstream output;
  output << "# panel_sha256=" << kLateGatePanelSha256 << '\n'
         << "# schema=" << kLateGatePlanSchema << '\n'
         << "# selection=" << kLateGateSelection << '\n'
         << "# source_sha256=" << std::string(64U, 'b') << '\n'
         << kLateGateHeader << '\n';
  for (const RenderedRoot &row : rows) {
    output << row.run << '\t' << row.population << '\t' << row.role << '\t'
           << std::string(64U, 'c') << '\t' << std::string(64U, 'd') << '\t'
           << row.game_id << '\t' << row.color << '\t' << row.depth << '\t'
           << row.state_id << '\t' << row.canonical_key << '\t'
           << row.transcript << '\n';
  }
  return output.str();
}

LateGatePlan parse_plan_text(const std::string &raw) {
  const std::string digest = native::sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(raw.data()), raw.size()));
  const auto path = std::filesystem::temp_directory_path() /
                    ("jacek-late-gate-plan-" + digest + ".tsv");
  {
    std::ofstream output(path, std::ios::binary);
    output << raw;
  }
  try {
    LateGatePlan result = read_late_gate_plan(path.string(), digest);
    std::filesystem::remove(path);
    return result;
  } catch (...) {
    std::filesystem::remove(path);
    throw;
  }
}

void require_plan_rejected(const std::string &raw, std::string_view message) {
  bool rejected = false;
  try {
    (void)parse_plan_text(raw);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, message);
}

std::string duplicate_first_source_game(const std::string &raw) {
  std::vector<std::string> lines;
  std::istringstream input(raw);
  for (std::string line; std::getline(input, line);) lines.push_back(line);
  const auto header = std::find(lines.begin(), lines.end(), kLateGateHeader);
  require(header != lines.end() && header + 2 < lines.end(),
          "synthetic plan must contain two rows");
  std::vector<std::string> first;
  for (const std::string_view field : split_exact(*(header + 1), '\t')) {
    first.emplace_back(field);
  }
  std::vector<std::string> second;
  for (const std::string_view field : split_exact(*(header + 2), '\t')) {
    second.emplace_back(field);
  }
  require(first.size() == 11U && second.size() == 11U,
          "synthetic rows must have eleven fields");
  second[5] = first[5];
  std::ostringstream rebuilt;
  for (auto line = lines.begin(); line != header + 2; ++line) {
    rebuilt << *line << '\n';
  }
  for (std::size_t index = 0; index < second.size(); ++index) {
    if (index) rebuilt << '\t';
    rebuilt << second[index];
  }
  rebuilt << '\n';
  for (auto line = header + 3; line != lines.end(); ++line) {
    rebuilt << *line << '\n';
  }
  return rebuilt.str();
}

void test_plan_replay_and_balance() {
  const std::string raw = make_valid_plan();
  const LateGatePlan parsed = parse_plan_text(raw);
  require(parsed.roots.size() == 64U, "valid gate plan must contain 64 roots");
  require(parsed.panel_sha256 == kLateGatePanelSha256,
          "gate plan must bind its panel");
  const std::string digest = native::sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(raw.data()), raw.size()));
  const auto path = std::filesystem::temp_directory_path() /
                    ("jacek-late-gate-plan-wrong-sha-" + digest + ".tsv");
  {
    std::ofstream output(path, std::ios::binary);
    output << raw;
  }
  bool rejected = false;
  try {
    (void)read_late_gate_plan(path.string(), std::string(64U, '0'));
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  std::filesystem::remove(path);
  require(rejected, "wrong plan SHA-256 must fail closed");
  require_plan_rejected(
      duplicate_first_source_game(raw),
      "duplicate source games must fail closed");
  require_plan_rejected(
      make_valid_plan(true),
      "compensated population imbalance must fail closed");
}

}  // namespace

int main() {
  try {
    test_plan_replay_and_balance();
  } catch (const std::exception &error) {
    std::cerr << "jacek native late-pacing game gate test failed: "
              << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
