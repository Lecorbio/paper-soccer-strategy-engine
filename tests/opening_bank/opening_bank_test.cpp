#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "opening_bank_internal.hpp"
#include "papersoccer/rules.hpp"

namespace opening_bank = papersoccer::opening_bank;

namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Function>
void require_invalid_argument(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

class TemporaryBankFile {
 public:
  explicit TemporaryBankFile(const std::string &contents) {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    path_ = std::filesystem::temp_directory_path() /
            ("papersoccer-opening-bank-test-" + std::to_string(nonce) +
             ".tsv");
    std::ofstream output(path_, std::ios::binary | std::ios::trunc);
    output << contents;
    if (!output) {
      throw std::runtime_error("could not create temporary opening bank");
    }
  }

  TemporaryBankFile(const TemporaryBankFile &) = delete;
  TemporaryBankFile &operator=(const TemporaryBankFile &) = delete;

  ~TemporaryBankFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

  const std::filesystem::path &path() const noexcept { return path_; }

 private:
  std::filesystem::path path_{};
};

std::vector<papersoccer::Move> reflected(
    const std::vector<papersoccer::Move> &moves) {
  std::vector<papersoccer::Move> result;
  result.reserve(moves.size());
  for (const papersoccer::Move move : moves) {
    result.push_back(papersoccer::Move{{8 - move.to.x, move.to.y}});
  }
  return result;
}

std::uint64_t splitmix_next(std::uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  std::uint64_t value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::size_t unbiased_index(std::uint64_t &state, std::size_t upper_bound) {
  const std::uint64_t bound = static_cast<std::uint64_t>(upper_bound);
  const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
  std::uint64_t value = 0;
  do {
    value = splitmix_next(state);
  } while (value < threshold);
  return static_cast<std::size_t>(value % bound);
}

papersoccer::GameState uniform_candidate(std::size_t depth,
                                         std::uint64_t generation_seed) {
  papersoccer::GameState state = papersoccer::make_initial_state();
  for (std::size_t ply = 0; ply < depth && !papersoccer::is_terminal(state);
       ++ply) {
    const std::vector<papersoccer::Move> legal = papersoccer::legal_moves(state);
    const papersoccer::Move selected =
        legal[unbiased_index(generation_seed, legal.size())];
    state = papersoccer::apply_move(state, selected);
  }
  return state;
}

opening_bank::Bank one_record_bank(const opening_bank::OpeningRecord &source,
                                   std::string phase,
                                   std::vector<papersoccer::Move> moves) {
  const papersoccer::GameState replayed = opening_bank::replay_transcript(moves);
  opening_bank::OpeningRecord record;
  record.phase = phase;
  record.depth = moves.size();
  record.generation_seed = source.generation_seed;
  record.state_hash = opening_bank::state_hash(replayed);
  record.canonical_key = opening_bank::canonical_key(replayed);
  record.to_move = replayed.to_move;
  record.moves = std::move(moves);
  record.opening_id = opening_bank::stable_opening_id(
      record.phase, record.depth, record.state_hash);
  return opening_bank::Bank{std::move(phase), record.depth, 1,
                            source.generation_seed, {std::move(record)}};
}

void generation_is_deterministic_and_replayable() {
  const std::string initial_digest =
      "da9c9aed1dff58b40324f331bc33f60fd929ab988cfdf5498bf8e060e34774f1";
  const papersoccer::GameState initial = papersoccer::make_initial_state();
  require(opening_bank::state_hash(initial) == initial_digest,
          "state hashing should match the standard SHA-256 test fixture");
  require(opening_bank::canonical_key(initial) == initial_digest,
          "the symmetric initial state should canonicalize to itself");

  constexpr std::uint64_t seed = 18'446'744'073'709'551'615ULL;
  const opening_bank::Bank first =
      opening_bank::generate_bank("development", 12, 16, seed);
  const opening_bank::Bank repeated =
      opening_bank::generate_bank("development", 12, 16, seed);
  require(first == repeated,
          "the same seed should reproduce identical opening records");

  const std::string rendered = opening_bank::render_bank(first);
  require(rendered == opening_bank::render_bank(repeated),
          "the same seed should reproduce byte-identical TSV");
  require(rendered.starts_with("schema\tpapersoccer.opening-bank.v1\n"),
          "the bank should start with the frozen schema line");
  require(rendered.find(
              "state_hash_algorithm\tsha256-canonical-game-state/v1\n") !=
              std::string::npos,
          "the bank should declare the preregistered SHA-256 algorithm");
  require(rendered.ends_with('\n'),
          "strict TSV output should end with a line feed");

  const opening_bank::Bank parsed =
      opening_bank::parse_bank_text(rendered, "deterministic-memory-bank");
  require(parsed == first, "rendered records should parse without information loss");

  TemporaryBankFile file(rendered);
  require(opening_bank::load_bank(file.path()) == first,
          "the file validator should replay and recover a valid bank");

  std::unordered_set<std::string> identifiers;
  std::unordered_set<std::string> states;
  std::unordered_set<std::string> canonical;
  for (const opening_bank::OpeningRecord &record : first.records) {
    require(record.moves.size() == first.depth,
            "each transcript should have exactly the requested physical plies");
    const papersoccer::GameState replayed =
        opening_bank::replay_transcript(record.moves);
    require(!papersoccer::is_terminal(replayed),
            "accepted openings should be non-terminal");
    require(replayed.path.size() == first.depth + 1,
            "each selected edge should add exactly one path point");
    require(replayed.used_segments.size() == first.depth,
            "each opening ply should be one distinct physical edge");
    require(record.state_hash == opening_bank::state_hash(replayed),
            "the state hash should reproduce from the transcript");
    require(record.canonical_key == opening_bank::canonical_key(replayed),
            "the canonical key should reproduce from the transcript");
    require(record.to_move == replayed.to_move,
            "the side to move should reproduce from the transcript");
    require(identifiers.insert(record.opening_id).second,
            "opening IDs should be unique within a bank");
    require(states.insert(record.state_hash).second,
            "logical states should be unique within a bank");
    require(canonical.insert(record.canonical_key).second,
            "horizontal equivalence classes should be unique within a bank");
  }
}

void terminals_and_tampered_contracts_are_rejected() {
  const std::vector<papersoccer::Move> terminal_transcript{
      {{4, 5}}, {{3, 5}}, {{4, 4}}, {{3, 3}},
      {{3, 2}}, {{4, 1}}, {{4, 0}},
  };
  require_invalid_argument(
      [&] { (void)opening_bank::replay_transcript(terminal_transcript); },
      "a transcript ending in a goal should be rejected");

  const std::string zero_hash(64, '0');
  const std::string terminal_bank =
      "schema\t" + std::string(opening_bank::kSchema) +
      "\nphase\tdevelopment\ndepth\t7\npairs\t1\nrules\t" +
      std::string(opening_bank::kRules) + "\ngenerator\t" +
      std::string(opening_bank::kGenerator) +
      "\ngenerator_seed\t1\nselection\t" +
      std::string(opening_bank::kSelection) +
      "\nstate_hash_algorithm\t" +
      std::string(opening_bank::kStateHashAlgorithm) +
      "\ncanonicalization\t" +
      std::string(opening_bank::kCanonicalization) +
      "\nopening_ply_definition\t" +
      std::string(opening_bank::kOpeningPlyDefinition) +
      "\nopening_id\tphase\tdepth\tgeneration_seed\tstate_hash\t"
      "canonical_key\tto_move\tmoves\n" +
      opening_bank::stable_opening_id("development", 7, zero_hash) +
      "\tdevelopment\t7\t1\t" + zero_hash + "\t" + zero_hash +
      "\tone\t4,5;3,5;4,4;3,3;3,2;4,1;4,0\n";
  require_invalid_argument(
      [&] {
        (void)opening_bank::parse_bank_text(terminal_bank,
                                            "terminal-bank-fixture");
      },
      "the bank validator should reject a terminal transcript");

  const opening_bank::Bank bank =
      opening_bank::generate_bank("development", 4, 2, 1234);
  std::string tampered = opening_bank::render_bank(bank);
  const std::string declared =
      "state_hash_algorithm\tsha256-canonical-game-state/v1";
  const std::size_t position = tampered.find(declared);
  require(position != std::string::npos,
          "the test fixture should contain the algorithm declaration");
  tampered.replace(position, declared.size(),
                   "state_hash_algorithm\tsha256-other/v1");
  require_invalid_argument(
      [&] { (void)opening_bank::parse_bank_text(tampered, "tampered-bank"); },
      "a changed state-hash contract should be rejected");
}

void terminal_generation_candidates_are_retried() {
  std::uint64_t candidate_seed_stream = 1;
  const std::uint64_t first_terminal_seed =
      splitmix_next(candidate_seed_stream);
  const std::uint64_t second_terminal_seed =
      splitmix_next(candidate_seed_stream);
  const std::uint64_t accepted_seed = splitmix_next(candidate_seed_stream);
  require(papersoccer::is_terminal(
              uniform_candidate(20, first_terminal_seed)) &&
              papersoccer::is_terminal(
                  uniform_candidate(20, second_terminal_seed)),
          "the frozen seed fixture should start with two terminal candidates");
  require(!papersoccer::is_terminal(uniform_candidate(20, accepted_seed)),
          "the third frozen candidate should be non-terminal");

  const opening_bank::Bank bank =
      opening_bank::generate_bank("development", 20, 1, 1);
  require(bank.records.front().generation_seed == accepted_seed,
          "generation should reject terminal candidates and advance twice");
}

void horizontal_reflections_are_one_equivalence_class() {
  const opening_bank::Bank generated =
      opening_bank::generate_bank("development", 8, 8, 98'765);
  const opening_bank::OpeningRecord *asymmetric = nullptr;
  opening_bank::Bank mirror;
  for (const opening_bank::OpeningRecord &record : generated.records) {
    opening_bank::Bank candidate = one_record_bank(
        record, "validation", reflected(record.moves));
    if (candidate.records.front().state_hash != record.state_hash) {
      asymmetric = &record;
      mirror = std::move(candidate);
      break;
    }
  }
  require(asymmetric != nullptr,
          "the deterministic sample should contain an asymmetric opening");
  require(mirror.records.front().canonical_key == asymmetric->canonical_key,
          "a horizontal reflection should retain the canonical key");

  const opening_bank::Bank original =
      one_record_bank(*asymmetric, "development", asymmetric->moves);
  require_invalid_argument(
      [&] { opening_bank::validate_disjoint({original, mirror}); },
      "cross-bank validation should reject horizontal reflections");
}

void exclusions_make_phase_banks_disjoint() {
  constexpr std::uint64_t seed = 42;
  const opening_bank::Bank development =
      opening_bank::generate_bank("development", 4, 5, seed);
  const opening_bank::Bank validation =
      opening_bank::generate_bank("validation", 4, 5, seed, {development});
  const opening_bank::Bank test = opening_bank::generate_bank(
      "test", 4, 5, seed, {development, validation});
  opening_bank::validate_disjoint({development, validation, test});

  require(validation.records.front().generation_seed !=
              development.records.front().generation_seed,
          "an excluded duplicate should advance the candidate seed stream");
  require(test.records.front().generation_seed !=
              development.records.front().generation_seed &&
              test.records.front().generation_seed !=
                  validation.records.front().generation_seed,
          "test generation should skip both prior phases");
}

}  // namespace

int main() {
  try {
    generation_is_deterministic_and_replayable();
    terminals_and_tampered_contracts_are_rejected();
    terminal_generation_candidates_are_retried();
    horizontal_reflections_are_one_equivalence_class();
    exclusions_make_phase_banks_disjoint();
    std::cout << "[PASS] opening bank test\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "[FAIL] opening bank test: " << error.what() << '\n';
    return 1;
  }
}
