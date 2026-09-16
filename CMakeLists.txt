cmake_minimum_required(VERSION 3.20)

project(papersoccer LANGUAGES CXX)

# CONFIGURE_DEPENDS notices the validation lock's absent-to-present transition.
# That forces a reconfigure and changes a core compile definition, so an
# incremental post-selection build cannot retain the pre-lock branch.
file(GLOB PAPERSOCCER_GAME_REVIEW_LOCK_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/game_review_gate/game_review_calibration_lock.hpp")
file(GLOB PAPERSOCCER_JACEK_NATIVE_ROUND2_DEPLOYMENTS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_native_round2_deployment.json")
file(GLOB_RECURSE PAPERSOCCER_JACEK_NATIVE_ROUND2_ACTIVATION_INPUTS
  CONFIGURE_DEPENDS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/models/*.json"
  "${CMAKE_CURRENT_SOURCE_DIR}/models/*.runtime")
file(GLOB_RECURSE PAPERSOCCER_JACEK_NATIVE_ROUND2_GATE_EVIDENCE
  CONFIGURE_DEPENDS LIST_DIRECTORIES false
  "${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_native_round2_gate_evidence/*")
if(PAPERSOCCER_GAME_REVIEW_LOCK_FILES)
  set(PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT 1)
else()
  set(PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT 0)
endif()

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE PAPERSOCCER_SOURCE_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
string(LENGTH "${PAPERSOCCER_SOURCE_COMMIT}"
  PAPERSOCCER_SOURCE_COMMIT_LENGTH)
if(NOT PAPERSOCCER_SOURCE_COMMIT_LENGTH EQUAL 40 OR
   NOT PAPERSOCCER_SOURCE_COMMIT MATCHES "^[0-9a-f]+$")
  set(PAPERSOCCER_SOURCE_COMMIT "unknown")
endif()
execute_process(
  COMMAND git status --porcelain --untracked-files=all
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE PAPERSOCCER_SOURCE_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)
if(PAPERSOCCER_SOURCE_STATUS STREQUAL "")
  set(PAPERSOCCER_SOURCE_DIRTY false)
else()
  set(PAPERSOCCER_SOURCE_DIRTY true)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(PAPERSOCCER_ENABLE_SANITIZERS
  "Build native targets with AddressSanitizer and UndefinedBehaviorSanitizer"
  OFF)
if(PAPERSOCCER_ENABLE_SANITIZERS)
  if(EMSCRIPTEN)
    message(FATAL_ERROR "PAPERSOCCER_ENABLE_SANITIZERS is for native builds only")
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    message(FATAL_ERROR
      "PAPERSOCCER_ENABLE_SANITIZERS requires GCC or a Clang-family compiler")
  endif()
  add_compile_options(
    -O1
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all)
  add_link_options(
    -fsanitize=address,undefined
    -fno-omit-frame-pointer
    -fno-sanitize-recover=all)
  set(PAPERSOCCER_SANITIZERS_ENABLED true)
else()
  set(PAPERSOCCER_SANITIZERS_ENABLED false)
endif()

add_library(papersoccer_core
  src/analysis/complete_turn_analysis.cpp
  src/analysis/exact_endgame_solver.cpp
  src/analysis/game_review_session.cpp
  src/bots/alpha_beta/alpha_beta.cpp
  src/bots/bot.cpp
  src/bots/jacek_inspired/jacek_features.cpp
  src/bots/jacek_inspired/jacek_inspired.cpp
  src/bots/jacek_inspired/jacek_neural.cpp
  src/bots/jacek_replay_bfm/features.cpp
  src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp
  src/bots/jacek_replay_bfm/model.cpp
  src/bots/mcts/mcts.cpp
  src/bots/mcts/tactical_search.cpp
  src/bots/rank5_derived/rank5_derived.cpp
  src/bots/random/random_bot.cpp
  src/core/debug.cpp
  src/core/geometry.cpp
  src/core/match.cpp
  src/core/rules.cpp
  src/web/web_game.cpp
)
target_include_directories(papersoccer_core PUBLIC include)
target_include_directories(papersoccer_core PRIVATE src/bots)
target_compile_definitions(papersoccer_core PRIVATE
  PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT=${PAPERSOCCER_GAME_REVIEW_LOCK_PRESENT})
if(EMSCRIPTEN)
  # Alpha-beta uses a private exception to unwind cleanly when its hard node
  # budget is reached. Every Wasm consumer must compile the core with catches
  # enabled, not only the test executable that links it.
  target_compile_options(papersoccer_core PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
endif()

add_executable(papersoccer_cli src/cli/main.cpp)
target_link_libraries(papersoccer_cli PRIVATE papersoccer_core)

add_executable(papersoccer_replay_export src/replay/main.cpp)
target_link_libraries(papersoccer_replay_export PRIVATE papersoccer_core)

add_library(papersoccer_arena_support
  src/arena/report.cpp
  src/arena/runner.cpp
)
target_include_directories(papersoccer_arena_support PUBLIC include)
target_link_libraries(papersoccer_arena_support PUBLIC papersoccer_core)
set(PAPERSOCCER_ARENA_CONFIGURED_FLAGS
  "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -std=c++20")
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  string(APPEND PAPERSOCCER_ARENA_CONFIGURED_FLAGS
    " -Wall -Wextra -Wpedantic")
endif()
if(PAPERSOCCER_ENABLE_SANITIZERS)
  string(APPEND PAPERSOCCER_ARENA_CONFIGURED_FLAGS
    " -O1 -fsanitize=address,undefined -fno-omit-frame-pointer"
    " -fno-sanitize-recover=all")
endif()
string(STRIP "${PAPERSOCCER_ARENA_CONFIGURED_FLAGS}"
  PAPERSOCCER_ARENA_CONFIGURED_FLAGS)
target_compile_definitions(papersoccer_arena_support PRIVATE
  PAPERSOCCER_BUILD_TYPE="$<CONFIG>"
  PAPERSOCCER_COMPILER_ID="${CMAKE_CXX_COMPILER_ID}"
  PAPERSOCCER_COMPILER_VERSION="${CMAKE_CXX_COMPILER_VERSION}"
  PAPERSOCCER_CONFIGURED_FLAGS="${PAPERSOCCER_ARENA_CONFIGURED_FLAGS}"
  PAPERSOCCER_SANITIZERS_ENABLED=${PAPERSOCCER_SANITIZERS_ENABLED}
  PAPERSOCCER_SOURCE_COMMIT="${PAPERSOCCER_SOURCE_COMMIT}"
  PAPERSOCCER_SOURCE_DIRTY=${PAPERSOCCER_SOURCE_DIRTY})

add_executable(papersoccer_arena src/arena/main.cpp)
target_link_libraries(papersoccer_arena PRIVATE papersoccer_arena_support)

if(EMSCRIPTEN)
  target_link_options(papersoccer_cli PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_replay_export PRIVATE
    -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_arena PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
endif()

add_executable(papersoccer_tests
  tests/analysis/game_review_test.cpp
  tests/bots/alpha_beta/alpha_beta_test.cpp
  tests/bots/jacek_inspired/jacek_inspired_test.cpp
  tests/bots/jacek_replay_bfm/jacek_replay_bfm_test.cpp
  tests/bots/mcts/mcts_test.cpp
  tests/bots/rank5_derived/rank5_derived_test.cpp
  tests/bots/random/random_bot_test.cpp
  tests/core/match_test.cpp
  tests/core/rules_test.cpp
  tests/test_main.cpp
  tests/web/web_game_session_test.cpp
)
target_link_libraries(papersoccer_tests PRIVATE papersoccer_core)
target_include_directories(papersoccer_tests PRIVATE src/bots)
if(EMSCRIPTEN)
  target_compile_definitions(papersoccer_tests PRIVATE
    PAPERSOCCER_SOURCE_DIR="")
  target_compile_options(papersoccer_tests PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_tests PRIVATE
    -sDISABLE_EXCEPTION_CATCHING=0
    --embed-file
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/rank5_derived_vs_mcts_path.txt@/tests/fixtures/rank5_derived_vs_mcts_path.txt")
else()
  target_compile_definitions(papersoccer_tests PRIVATE
    PAPERSOCCER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
endif()

enable_testing()
add_test(NAME papersoccer_tests COMMAND papersoccer_tests)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResearchChecks.cmake)
# Keep the public bot registry and exact submission compilation routes here.
# Existing leaderboard/provenance readers bind these declarations to this file.
set(PAPERSOCCER_CODINGAME_BOTS
  alpha_beta rank_5 rank_4 rank_4_jacek_hybrid rank_4_fullturn_bfm rank_4_exchange challenger selfplay_nn jacek_nn
  jacek_native_bfm jacek_arena_bfm
  neural_puct topology
  safe_inward rebound_proof defensive_proof shell_proof all_depth_proof
  reply_proof exchange_proof frontier_proof conservative_frontier_proof)

# Research-only entrants are built and tested but are deliberately outside the
# frozen published leaderboard roster until a separate leaderboard campaign
# reviews them.  Their own campaign receipts must not imply a leaderboard claim.
set(PAPERSOCCER_CODINGAME_RESEARCH_BOTS
  compact_value_bfm)
set(PAPERSOCCER_CODINGAME_BUILD_BOTS
  ${PAPERSOCCER_CODINGAME_BOTS}
  ${PAPERSOCCER_CODINGAME_RESEARCH_BOTS})

macro(papersoccer_register_codingame_submissions)
  foreach(PAPERSOCCER_CODINGAME_BOT IN LISTS PAPERSOCCER_CODINGAME_BUILD_BOTS)
    papersoccer_codingame_target_prefix(
      PAPERSOCCER_CODINGAME_PREFIX "${PAPERSOCCER_CODINGAME_BOT}")
    set(PAPERSOCCER_CODINGAME_BOT_DIRECTORY
      "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/${PAPERSOCCER_CODINGAME_BOT}")

    add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_submission
      "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/submission.cpp")
    add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_submission_test
      "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/submission_test.cpp")
    if(PAPERSOCCER_CODINGAME_BOT STREQUAL "all_depth_proof" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "reply_proof" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "exchange_proof" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "frontier_proof" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "conservative_frontier_proof" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "rank_4_fullturn_bfm" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_native_bfm" OR
       PAPERSOCCER_CODINGAME_BOT STREQUAL "rank_4_exchange")
      target_include_directories(${PAPERSOCCER_CODINGAME_PREFIX}_submission_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
      target_link_libraries(${PAPERSOCCER_CODINGAME_PREFIX}_submission_test
        PRIVATE papersoccer_core)
    endif()
    add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test
             COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test)
    if(PAPERSOCCER_CODINGAME_BOT STREQUAL "compact_value_bfm")
      add_executable(
        ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_feature_reference
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/submission_test.cpp")
      target_compile_definitions(
        ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_feature_reference
        PRIVATE COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT=1)
      add_test(
        NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_feature_reference
        COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_feature_reference)

      add_executable(
        ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_selection_reference
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/submission_test.cpp")
      target_compile_definitions(
        ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_selection_reference
        PRIVATE COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT=1)
      add_test(
        NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_selection_reference
        COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_selection_reference)

      add_executable(
        ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_search_baseline
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/submission_test.cpp")
      target_compile_definitions(
        ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_search_baseline PRIVATE
        COMPACT_VALUE_BFM_REFERENCE_FEATURE_SORT=1
        COMPACT_VALUE_BFM_REFERENCE_DESCENDANT_SORT=1)
      add_test(
        NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_search_baseline
        COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test_search_baseline)
    endif()

    if(EXISTS "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/timing_probe.cpp")
      add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/timing_probe.cpp")
      if(PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_native_bfm" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_arena_bfm" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "compact_value_bfm")
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe 0)
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe_player1
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe 1)
        set_tests_properties(
          ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
          ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe_player1
          PROPERTIES RUN_SERIAL TRUE)
        if(PAPERSOCCER_SANITIZERS_ENABLED)
          set_tests_properties(
            ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
            ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe_player1
            PROPERTIES DISABLED TRUE)
        endif()
      else()
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
                 COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe)
      endif()
    endif()

    if(EXISTS "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/comparison_gate.cpp")
      add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/comparison_gate.cpp")
      if(PAPERSOCCER_CODINGAME_BOT STREQUAL "rank_4_jacek_hybrid")
        target_sources(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate PRIVATE
          "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/comparison_gate_rank4.cpp"
          "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/comparison_gate_hybrid.cpp")
        target_include_directories(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/opening_bank")
        target_link_libraries(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
          PRIVATE papersoccer_opening_bank_support)
      endif()
      if(NOT PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_arena_bfm" AND
         NOT PAPERSOCCER_CODINGAME_BOT STREQUAL "compact_value_bfm")
        target_include_directories(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
        target_link_libraries(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
          PRIVATE papersoccer_core)
      endif()
      if(PAPERSOCCER_CODINGAME_BOT STREQUAL "rank_4_jacek_hybrid")
        set(PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY
          results/rank_4_jacek_hybrid/openings)
        set(PAPERSOCCER_HYBRID_GATE_FIXTURE_ROOT
          "${CMAKE_CURRENT_BINARY_DIR}/rank_4_jacek_hybrid_comparison_gate_fixture")
        set(PAPERSOCCER_HYBRID_GATE_FIXTURE_BANK_DIRECTORY
          "${PAPERSOCCER_HYBRID_GATE_FIXTURE_ROOT}/${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}")
        file(MAKE_DIRECTORY
          "${PAPERSOCCER_HYBRID_GATE_FIXTURE_BANK_DIRECTORY}")
        foreach(PAPERSOCCER_HYBRID_GATE_BANK_NAME IN ITEMS
            development_d04.tsv
            development_d08.tsv
            development_d12.tsv
            development_d20.tsv)
          set(PAPERSOCCER_HYBRID_GATE_BANK_SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}/${PAPERSOCCER_HYBRID_GATE_BANK_NAME}")
          set(PAPERSOCCER_HYBRID_GATE_BANK_FIXTURE
            "${PAPERSOCCER_HYBRID_GATE_FIXTURE_BANK_DIRECTORY}/${PAPERSOCCER_HYBRID_GATE_BANK_NAME}")
          file(REMOVE "${PAPERSOCCER_HYBRID_GATE_BANK_FIXTURE}")
          configure_file(
            "${PAPERSOCCER_HYBRID_GATE_BANK_SOURCE}"
            "${PAPERSOCCER_HYBRID_GATE_BANK_FIXTURE}"
            COPYONLY)
          file(CHMOD "${PAPERSOCCER_HYBRID_GATE_BANK_FIXTURE}"
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ)
        endforeach()
        set(PAPERSOCCER_HYBRID_GATE_BANK_ARGS
          --bank ${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}/development_d04.tsv
          --bank ${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}/development_d08.tsv
          --bank ${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}/development_d12.tsv
          --bank ${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}/development_d20.tsv
          --expected-role development
          --expected-seeds 18128950407139886133,9297997631523997120,11025886481058993262,4624785204876369057
          --expected-sha256 984fbb78d85d7f9806c77e675b9b22a9b047bd15311f510ab0cedcd9a63244dc,6dbec157e7094f07796a9aa1ac97b43919930377ec315c761eaace216630259e,d30d087020e4946ce77b6d6e578484d583f0cf25f2ffa90a1918f8d9a9a8a11a,2aa4b635dcaf23b2587b22fdb7558f4c8d6b4dd5a33e3fec2c164931b3fcd8d4
          --expected-depths 4,8,12,20)
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_self_test
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            ${PAPERSOCCER_HYBRID_GATE_BANK_ARGS} --self-test)
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_sha
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --bank ${PAPERSOCCER_HYBRID_GATE_BANK_DIRECTORY}/development_d04.tsv
            --expected-role development --expected-depths 4
            --expected-seeds 18128950407139886133
            --expected-sha256 0000000000000000000000000000000000000000000000000000000000000000
            --self-test)
        set_tests_properties(
          ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_sha
          PROPERTIES WILL_FAIL TRUE)
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            ${PAPERSOCCER_HYBRID_GATE_BANK_ARGS}
            --profile nodes --candidate-nodes 500 --reference-nodes 500
            --max-turns 320 --candidate-exact-proof-mask 0)
        set_tests_properties(
          ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_self_test
          ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_sha
          ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          PROPERTIES WORKING_DIRECTORY
            "${PAPERSOCCER_HYBRID_GATE_FIXTURE_ROOT}")
      elseif(PAPERSOCCER_CODINGAME_BOT STREQUAL "neural_puct")
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --candidate-simulations 32 --candidate-nodes 2048
            --incumbent-nodes 200 --opening-limit 1 --max-turns 320
            --output
            ${CMAKE_CURRENT_BINARY_DIR}/neural-puct-comparison-smoke.json)
      elseif(PAPERSOCCER_CODINGAME_BOT STREQUAL "rank_4_fullturn_bfm")
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --pairs-per-depth 1 --candidate-work 2000
            --reference-nodes 2000 --max-turns 200
            --opening-turns 0,1,2,3)
      elseif(PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_native_bfm")
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --pairs 1 --candidate-iterations 2
            --candidate-actions 16 --candidate-partial-paths 500
            --candidate-tree-nodes 128 --reference-nodes 200
            --maximum-turns 384 --opening-turns 0)
        add_test(
          NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_unfinished
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --pairs 1 --candidate-iterations 1
            --candidate-actions 8 --candidate-partial-paths 32
            --candidate-tree-nodes 32 --reference-nodes 10
            --maximum-turns 1 --opening-turns 0)
        set_tests_properties(
          ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_unfinished
          PROPERTIES WILL_FAIL TRUE)
        add_test(
          NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_score
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --pairs 1 --candidate-iterations 2
            --candidate-actions 16 --candidate-partial-paths 500
            --candidate-tree-nodes 128 --reference-nodes 200
            --maximum-turns 384 --opening-turns 0
            --minimum-wilson-lower 1.0)
        set_tests_properties(
          ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_rejects_score
          PROPERTIES WILL_FAIL TRUE)
      elseif(PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_arena_bfm" OR
             PAPERSOCCER_CODINGAME_BOT STREQUAL "compact_value_bfm")
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate)
      elseif(PAPERSOCCER_CODINGAME_BOT STREQUAL "topology" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "safe_inward" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "rebound_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "defensive_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "shell_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "all_depth_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "reply_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "exchange_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "frontier_proof" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "conservative_frontier_proof")
        if(PAPERSOCCER_CODINGAME_BOT STREQUAL "topology")
          set(PAPERSOCCER_CODINGAME_SMOKE_BANK
            ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/promotion/reference/rank1_validation.tsv)
        else()
          set(PAPERSOCCER_CODINGAME_SMOKE_BANK
            ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/promotion/openings/initial.tsv)
        endif()
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
            --bank
            ${PAPERSOCCER_CODINGAME_SMOKE_BANK}
            --stage smoke --node-budget 500 --max-turns 320
            --shard-count 1 --shard-index 0
            --output
            ${CMAKE_CURRENT_BINARY_DIR}/${PAPERSOCCER_CODINGAME_BOT}-comparison-smoke.json)
      else()
        add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
          COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate 1 2000 200)
      endif()
    endif()
  endforeach()
endmacro()

macro(papersoccer_register_compact_candidate_checks)
  include(cmake/CompactCandidateChecks.cmake)
endmacro()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodingameTargets.cmake)
macro(papersoccer_register_submission_protocol_checks)
      foreach(PAPERSOCCER_CODINGAME_BOT IN LISTS PAPERSOCCER_CODINGAME_BUILD_BOTS)
        papersoccer_codingame_target_prefix(
          PAPERSOCCER_CODINGAME_PREFIX "${PAPERSOCCER_CODINGAME_BOT}")
        if(PAPERSOCCER_CODINGAME_BOT STREQUAL "compact_value_bfm" AND
           Python3_Interpreter_FOUND)
          add_test(
            NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_current
            COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/compact_value_bfm/export_submission.py
                    --check)
        else()
          add_test(
            NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_current
            COMMAND ${PAPERSOCCER_NODE_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/generate_submission.mjs
                    ${PAPERSOCCER_CODINGAME_BOT}
                    --check)
        endif()
        add_test(
          NAME ${PAPERSOCCER_CODINGAME_PREFIX}_protocol_smoke_test
          COMMAND ${PAPERSOCCER_NODE_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/protocol_smoke_test.mjs
                  $<TARGET_FILE:${PAPERSOCCER_CODINGAME_PREFIX}_submission>
        )
      endforeach()
endmacro()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/WebChecks.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/WasmTargets.cmake)
