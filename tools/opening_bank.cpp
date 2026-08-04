#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include "opening_bank_internal.hpp"

namespace opening_bank = papersoccer::opening_bank;

namespace {

enum class Command { Generate, Validate };

struct Options {
  Command command{Command::Generate};
  std::string phase{};
  std::size_t depth{};
  std::size_t pairs{};
  std::uint64_t seed{};
  bool phase_seen{};
  bool depth_seen{};
  bool pairs_seen{};
  bool seed_seen{};
  std::vector<std::filesystem::path> banks{};
  std::vector<std::filesystem::path> exclusions{};
};

void print_usage(std::ostream &out) {
  out << "Usage:\n"
      << "  papersoccer_opening_bank [generate] --phase "
         "development|validation|test --depth N --pairs N --seed UINT64 "
         "[--exclude-bank PATH ...]\n"
      << "  papersoccer_opening_bank validate --bank PATH "
         "[--bank PATH ...] [--exclude-bank PATH ...]\n";
}

std::string_view require_value(int argc, char **argv, int &index,
                               std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return argv[++index];
}

template <typename UInt>
UInt parse_unsigned(std::string_view text, std::string_view option) {
  static_assert(std::is_unsigned_v<UInt>);
  if (text.empty() || text.front() == '-') {
    throw std::invalid_argument(std::string(option) +
                                " requires an unsigned decimal integer");
  }
  UInt value{};
  const auto result =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument(std::string(option) +
                                " requires an unsigned decimal integer");
  }
  return value;
}

void reject_duplicate(bool &seen, std::string_view option) {
  if (seen) {
    throw std::invalid_argument(std::string(option) +
                                " may be specified only once");
  }
  seen = true;
}

Options parse_options(int argc, char **argv) {
  Options options;
  int index = 1;
  if (index < argc && std::string_view(argv[index]) == "generate") {
    ++index;
  } else if (index < argc && std::string_view(argv[index]) == "validate") {
    options.command = Command::Validate;
    ++index;
  }

  for (; index < argc; ++index) {
    const std::string_view argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      print_usage(std::cout);
      std::exit(0);
    }
    if (argument == "--phase") {
      reject_duplicate(options.phase_seen, argument);
      options.phase = require_value(argc, argv, index, argument);
    } else if (argument == "--depth") {
      reject_duplicate(options.depth_seen, argument);
      options.depth = parse_unsigned<std::size_t>(
          require_value(argc, argv, index, argument), argument);
    } else if (argument == "--pairs") {
      reject_duplicate(options.pairs_seen, argument);
      options.pairs = parse_unsigned<std::size_t>(
          require_value(argc, argv, index, argument), argument);
    } else if (argument == "--seed") {
      reject_duplicate(options.seed_seen, argument);
      options.seed = parse_unsigned<std::uint64_t>(
          require_value(argc, argv, index, argument), argument);
    } else if (argument == "--bank") {
      options.banks.emplace_back(require_value(argc, argv, index, argument));
    } else if (argument == "--exclude-bank") {
      options.exclusions.emplace_back(
          require_value(argc, argv, index, argument));
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }

  if (options.command == Command::Generate) {
    if (!options.phase_seen || !options.depth_seen || !options.pairs_seen ||
        !options.seed_seen) {
      throw std::invalid_argument(
          "generation requires --phase, --depth, --pairs, and --seed");
    }
    if (!options.banks.empty()) {
      throw std::invalid_argument("--bank is valid only with validate");
    }
  } else {
    if (options.banks.empty()) {
      throw std::invalid_argument("validation requires at least one --bank");
    }
    if (options.phase_seen || options.depth_seen || options.pairs_seen ||
        options.seed_seen) {
      throw std::invalid_argument(
          "generation options are not valid with validate");
    }
  }
  return options;
}

std::vector<opening_bank::Bank> load_all(
    const std::vector<std::filesystem::path> &paths) {
  std::vector<opening_bank::Bank> banks;
  banks.reserve(paths.size());
  for (const std::filesystem::path &path : paths) {
    banks.push_back(opening_bank::load_bank(path));
  }
  return banks;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    std::vector<opening_bank::Bank> excluded = load_all(options.exclusions);
    if (options.command == Command::Generate) {
      const opening_bank::Bank bank = opening_bank::generate_bank(
          options.phase, options.depth, options.pairs, options.seed, excluded);
      const std::string output = opening_bank::render_bank(bank);
      std::cout << output;
      if (!std::cout) {
        throw std::runtime_error("could not write opening bank to stdout");
      }
      return 0;
    }

    std::vector<opening_bank::Bank> banks = load_all(options.banks);
    banks.insert(banks.end(), excluded.begin(), excluded.end());
    opening_bank::validate_disjoint(banks);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "papersoccer_opening_bank: " << error.what() << '\n';
    print_usage(std::cerr);
    return 2;
  }
}
