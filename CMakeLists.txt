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
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CodingameTargets.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/WebChecks.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/WasmTargets.cmake)
