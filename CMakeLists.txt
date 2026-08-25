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

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND NOT EMSCRIPTEN)
  add_test(
    NAME papersoccer_jacek_model_current
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_jacek_neural_model.py
            --check
  )
  add_test(
    NAME papersoccer_flagship_study_unit_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/flagship_study
            -p test_*.py
  )
  set_tests_properties(papersoccer_flagship_study_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  add_test(
    NAME papersoccer_game_review_gate_unit_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/game_review_gate
            -p test_*.py
  )
  set_tests_properties(papersoccer_game_review_gate_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME papersoccer_codingame_promotion_unit_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame
            -p test_*.py)
  set_tests_properties(papersoccer_codingame_promotion_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME papersoccer_codingame_leaderboard_unit_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest discover
            -s ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame_leaderboard
            -p test_*.py)
  set_tests_properties(papersoccer_codingame_leaderboard_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME papersoccer_codingame_jacek_native_bfm_purity
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/jacek_native_bfm/check_purity.py)
  set_tests_properties(
    papersoccer_codingame_jacek_native_bfm_purity PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME papersoccer_codingame_jacek_native_restart_round2_unit_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest
            tests.codingame.test_jacek_native_restart_round2)
  set_tests_properties(
    papersoccer_codingame_jacek_native_restart_round2_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME papersoccer_codingame_jacek_native_late_pacing_gate_unit_tests
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest
            tests.codingame.test_jacek_native_late_pacing_106_gate
            tests.codingame.test_jacek_native_late_pacing_eval_panel
            tests.codingame.test_jacek_native_late_pacing_game_gate)
  set_tests_properties(
    papersoccer_codingame_jacek_native_late_pacing_gate_unit_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE PAPERSOCCER_JACEK_NATIVE_NUMPY_STATUS
    OUTPUT_QUIET
    ERROR_QUIET)
  if(PAPERSOCCER_JACEK_NATIVE_NUMPY_STATUS EQUAL 0)
    add_test(
      NAME papersoccer_codingame_jacek_native_training_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_jacek_native_training)
    set_tests_properties(
      papersoccer_codingame_jacek_native_training_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
      NAME papersoccer_codingame_jacek_native_round2_training_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_jacek_native_round2_training)
    set_tests_properties(
      papersoccer_codingame_jacek_native_round2_training_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
      NAME papersoccer_codingame_jacek_native_round2_selection_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_jacek_native_round2_selection)
    set_tests_properties(
      papersoccer_codingame_jacek_native_round2_selection_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
      NAME papersoccer_codingame_jacek_native_round2_activation_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_jacek_native_round2_activation)
    set_tests_properties(
      papersoccer_codingame_jacek_native_round2_activation_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
      NAME papersoccer_jacek_replay_training_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_jacek_replay_features
              tests.codingame.test_jacek_replay_corpus
              tests.codingame.test_jacek_replay_training
              tests.codingame.test_jacek_replay_promotion
              tests.codingame.test_jacek_replay_workflow
              tests.codingame.test_jacek_selfsearch_workflow
              tests.codingame.test_jacek_replay_baseline
              tests.codingame.test_jacek_replay_tuning)
    set_tests_properties(
      papersoccer_jacek_replay_training_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(
      NAME papersoccer_jacek_selfsearch_workflow_acceptance_smoke
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_jacek_selfsearch_acceptance)
    set_tests_properties(
      papersoccer_jacek_selfsearch_workflow_acceptance_smoke PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      TIMEOUT 120
      LABELS "selfsearch;acceptance")
  endif()

  add_test(
    NAME papersoccer_codingame_promotion_banks_current
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/promotion/build_goal_shell_banks.py
            --check)
  set_tests_properties(papersoccer_codingame_promotion_banks_current PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

  add_test(
    NAME papersoccer_codingame_promotion_manifest_valid
    COMMAND ${CMAKE_COMMAND} -E env
            PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/promotion_gate.py
            validate --bot all_depth_proof)
  set_tests_properties(papersoccer_codingame_promotion_manifest_valid PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set(PAPERSOCCER_CODINGAME_BOTS
  alpha_beta rank_5 rank_4 rank_4_jacek_hybrid rank_4_fullturn_bfm rank_4_exchange challenger selfplay_nn jacek_nn
  jacek_native_bfm jacek_arena_bfm
  neural_puct topology
  safe_inward rebound_proof defensive_proof shell_proof all_depth_proof
  reply_proof exchange_proof frontier_proof conservative_frontier_proof)

function(papersoccer_codingame_target_prefix output_variable bot_name)
  if(bot_name STREQUAL "alpha_beta")
    set(prefix papersoccer_codingame)
  else()
    set(prefix "papersoccer_codingame_${bot_name}")
  endif()
  set(${output_variable} "${prefix}" PARENT_SCOPE)
endfunction()

if(NOT EMSCRIPTEN)
  add_library(papersoccer_codingame_referee_support
    src/codingame/referee.cpp)
  target_link_libraries(papersoccer_codingame_referee_support
    PUBLIC papersoccer_core)
  target_include_directories(papersoccer_codingame_referee_support
    PUBLIC include)

  add_executable(papersoccer_codingame_referee
    src/codingame/main.cpp)
  target_link_libraries(papersoccer_codingame_referee
    PRIVATE papersoccer_codingame_referee_support)

  add_executable(papersoccer_codingame_referee_fake_bot
    tests/codingame/fake_bot.cpp)
  target_link_libraries(papersoccer_codingame_referee_fake_bot
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_referee_test
    tests/codingame/referee_test.cpp)
  target_link_libraries(papersoccer_codingame_referee_test
    PRIVATE papersoccer_codingame_referee_support)
  target_compile_definitions(papersoccer_codingame_referee_test PRIVATE
    PAPERSOCCER_CODINGAME_FAKE_BOT="$<TARGET_FILE:papersoccer_codingame_referee_fake_bot>"
    PAPERSOCCER_SANITIZERS_ENABLED=$<BOOL:${PAPERSOCCER_ENABLE_SANITIZERS}>)
  add_dependencies(papersoccer_codingame_referee_test
    papersoccer_codingame_referee_fake_bot)
  add_test(NAME papersoccer_codingame_referee_test
    COMMAND papersoccer_codingame_referee_test)
  set_tests_properties(papersoccer_codingame_referee_test PROPERTIES TIMEOUT 30)

  add_library(papersoccer_opening_bank_support
    src/opening_bank/opening_bank.cpp)
  target_include_directories(papersoccer_opening_bank_support
    PRIVATE src/opening_bank)
  target_link_libraries(papersoccer_opening_bank_support
    PUBLIC papersoccer_core)

  target_include_directories(papersoccer_arena
    PRIVATE src/opening_bank)
  target_link_libraries(papersoccer_arena
    PRIVATE papersoccer_opening_bank_support)

  add_executable(papersoccer_opening_bank
    tools/opening_bank.cpp)
  target_include_directories(papersoccer_opening_bank
    PRIVATE src/opening_bank)
  target_link_libraries(papersoccer_opening_bank
    PRIVATE papersoccer_opening_bank_support)

  add_executable(papersoccer_opening_bank_test
    tests/opening_bank/opening_bank_test.cpp)
  target_include_directories(papersoccer_opening_bank_test
    PRIVATE src/opening_bank)
  target_link_libraries(papersoccer_opening_bank_test
    PRIVATE papersoccer_opening_bank_support)
  add_test(NAME papersoccer_opening_bank_test
    COMMAND papersoccer_opening_bank_test)

  if(Python3_Interpreter_FOUND)
    add_test(
      NAME papersoccer_flagship_study_smoke
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/flagship_study/run_smoke.py
              --arena $<TARGET_FILE:papersoccer_arena>
              --manifest
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/flagship_study/smoke_manifest.json)
    set_tests_properties(papersoccer_flagship_study_smoke PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  add_executable(papersoccer_complete_turn_analysis_regression
    tools/rank5_derived_evaluator_gate.cpp)
  target_link_libraries(papersoccer_complete_turn_analysis_regression
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_complete_turn_analysis_regression_smoke
    COMMAND papersoccer_complete_turn_analysis_regression
            --nodes 500
            --depth 2
            --pairs-per-stage 1
            --bootstrap 100
            --max-plies 1024)

  add_executable(papersoccer_fullturn_replay_decision_auditor
    submissions/codingame/bots/rank_4_fullturn_bfm/replay_decision_auditor.cpp)
  target_include_directories(papersoccer_fullturn_replay_decision_auditor
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_fullturn_replay_decision_auditor
    PRIVATE papersoccer_core)

  add_executable(papersoccer_fullturn_replay_decision_auditor_test
    submissions/codingame/bots/rank_4_fullturn_bfm/replay_decision_auditor_test.cpp)
  target_include_directories(papersoccer_fullturn_replay_decision_auditor_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_fullturn_replay_decision_auditor_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_fullturn_replay_decision_auditor_test
    COMMAND papersoccer_fullturn_replay_decision_auditor_test)

  add_executable(papersoccer_arena_smoke_test
    tests/arena/arena_smoke_test.cpp)
  target_link_libraries(papersoccer_arena_smoke_test
    PRIVATE papersoccer_arena_support)
  add_test(NAME papersoccer_arena_smoke_test
           COMMAND papersoccer_arena_smoke_test)

  foreach(PAPERSOCCER_CODINGAME_BOT IN LISTS PAPERSOCCER_CODINGAME_BOTS)
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

    if(EXISTS "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/timing_probe.cpp")
      add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/timing_probe.cpp")
      if(PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_native_bfm" OR
         PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_arena_bfm")
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
      if(NOT PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_arena_bfm")
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
      elseif(PAPERSOCCER_CODINGAME_BOT STREQUAL "jacek_arena_bfm")
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

  add_executable(
    papersoccer_codingame_rank_4_jacek_hybrid_null_fastpath_comparison_gate
    submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_null_fastpath.cpp
    submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_hybrid.cpp
    submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_prefastpath.cpp)
  target_include_directories(
    papersoccer_codingame_rank_4_jacek_hybrid_null_fastpath_comparison_gate
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/rank_4_jacek_hybrid"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/bots"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/opening_bank")
  target_link_libraries(
    papersoccer_codingame_rank_4_jacek_hybrid_null_fastpath_comparison_gate
    PRIVATE papersoccer_opening_bank_support)

  add_executable(
    papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate
    submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_heldout.cpp
    submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_hybrid.cpp
    submissions/codingame/bots/rank_4_jacek_hybrid/comparison_gate_rank4.cpp)
  target_include_directories(
    papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src/bots"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/opening_bank")
  target_link_libraries(
    papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate
    PRIVATE papersoccer_opening_bank_support papersoccer_core)
  add_test(
    NAME
      papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate_self_test
    COMMAND
      papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate
      --bank
        results/rank_4_jacek_hybrid/openings/development_d04.tsv
      --expected-role development --expected-depths 4
      --expected-seeds 18128950407139886133
      --expected-sha256
        984fbb78d85d7f9806c77e675b9b22a9b047bd15311f510ab0cedcd9a63244dc
      --self-test)
  set_tests_properties(
    papersoccer_codingame_rank_4_jacek_hybrid_heldout_comparison_gate_self_test
    PROPERTIES
      TIMEOUT 30
      WORKING_DIRECTORY "${PAPERSOCCER_HYBRID_GATE_FIXTURE_ROOT}")

  add_executable(
    papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit
    submissions/codingame/bots/rank_4_jacek_hybrid/replay_tactical_audit.cpp)
  target_include_directories(
    papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/rank_4_jacek_hybrid"
      "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit
    PRIVATE papersoccer_core)
  add_test(
    NAME papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit_cheap
    COMMAND papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit
      --cheap)
  add_test(
    NAME papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit_full
    COMMAND papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit
      --full --state-cap 2000000 --global-state-cap 100000000)
  set_tests_properties(
    papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit_cheap
    PROPERTIES TIMEOUT 30)
  set_tests_properties(
    papersoccer_codingame_rank_4_jacek_hybrid_replay_tactical_audit_full
    PROPERTIES LABELS "qualification;long" RUN_SERIAL TRUE TIMEOUT 120)

  add_executable(papersoccer_codingame_rank_4_jacek_hybrid_parity_test
    submissions/codingame/bots/rank_4_jacek_hybrid/parity_test.cpp)
  target_include_directories(
    papersoccer_codingame_rank_4_jacek_hybrid_parity_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_rank_4_jacek_hybrid_parity_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_codingame_rank_4_jacek_hybrid_parity_test
    COMMAND papersoccer_codingame_rank_4_jacek_hybrid_parity_test)

  add_executable(papersoccer_codingame_jacek_arena_bfm_generator_benchmark
    submissions/codingame/bots/jacek_arena_bfm/generator_benchmark.cpp)
  add_executable(papersoccer_codingame_jacek_arena_bfm_selfplay_generator
    submissions/codingame/bots/jacek_arena_bfm/selfplay_generator.cpp)

  if(Python3_Interpreter_FOUND)
    add_test(NAME papersoccer_codingame_jacek_arena_bfm_purity
      COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/jacek_arena_bfm/check_purity.py)
    set_tests_properties(papersoccer_codingame_jacek_arena_bfm_purity
      PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  add_executable(papersoccer_codingame_neural_puct_feature_probe
    submissions/codingame/bots/neural_puct/feature_probe.cpp)

  add_executable(papersoccer_codingame_neural_puct_selfplay_generator
    submissions/codingame/bots/neural_puct/selfplay_generator.cpp)
  target_include_directories(
    papersoccer_codingame_neural_puct_selfplay_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_neural_puct_selfplay_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_neural_puct_live_replay_relabeler
    submissions/codingame/bots/neural_puct/live_replay_relabeler.cpp)
  target_include_directories(
    papersoccer_codingame_neural_puct_live_replay_relabeler
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_neural_puct_live_replay_relabeler
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_selfplay_nn_selfplay_generator
    submissions/codingame/bots/selfplay_nn/selfplay_generator.cpp)
  target_include_directories(
    papersoccer_codingame_selfplay_nn_selfplay_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_selfplay_nn_selfplay_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_rank_4_teacher_sample_generator
    submissions/codingame/bots/rank_4/teacher_sample_generator.cpp)
  target_include_directories(
    papersoccer_codingame_rank_4_teacher_sample_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_rank_4_teacher_sample_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_jacek_nn_teacher_sample_generator
    submissions/codingame/bots/jacek_nn/teacher_sample_generator.cpp)
  target_include_directories(
    papersoccer_codingame_jacek_nn_teacher_sample_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_jacek_nn_teacher_sample_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_replay_teacher
    tools/jacek_replay_bfm_teacher.cpp)
  target_include_directories(papersoccer_jacek_replay_teacher
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_replay_teacher
    PRIVATE papersoccer_core)

  function(papersoccer_hash_source_closure OUTPUT_VARIABLE)
    set(PAPERSOCCER_SOURCE_CLOSURE_MATERIAL "")
    foreach(PAPERSOCCER_SOURCE_CLOSURE_PATH IN ITEMS ${ARGN})
      file(SHA256
        "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_SOURCE_CLOSURE_PATH}"
        PAPERSOCCER_SOURCE_CLOSURE_FILE_SHA256)
      string(APPEND PAPERSOCCER_SOURCE_CLOSURE_MATERIAL
        "${PAPERSOCCER_SOURCE_CLOSURE_PATH}:${PAPERSOCCER_SOURCE_CLOSURE_FILE_SHA256}\n")
    endforeach()
    string(SHA256 PAPERSOCCER_SOURCE_CLOSURE_SHA256
      "${PAPERSOCCER_SOURCE_CLOSURE_MATERIAL}")
    set(${OUTPUT_VARIABLE} "${PAPERSOCCER_SOURCE_CLOSURE_SHA256}"
      PARENT_SCOPE)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ARGN})
  endfunction()

  set(PAPERSOCCER_SELFSEARCH_RANK4_ACTOR_SOURCE_PATHS
    include/papersoccer/types.hpp
    include/papersoccer/geometry.hpp
    include/papersoccer/rules.hpp
    src/bots/mcts_internal.hpp
    src/core/geometry.cpp
    src/core/rules.cpp
    submissions/codingame/bots/rank_4/replay_book.hpp
    submissions/codingame/bots/rank_4/replay_value_model.hpp
    submissions/codingame/bots/rank_4/teacher_residual_model.hpp
    submissions/codingame/bots/rank_4/bot.cpp)
  papersoccer_hash_source_closure(
    PAPERSOCCER_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256
    ${PAPERSOCCER_SELFSEARCH_RANK4_ACTOR_SOURCE_PATHS})

  set(PAPERSOCCER_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_PATHS
    include/papersoccer/types.hpp
    include/papersoccer/geometry.hpp
    include/papersoccer/rules.hpp
    src/bots/mcts_internal.hpp
    src/core/geometry.cpp
    src/core/rules.cpp
    submissions/codingame/bots/jacek_nn/replay_book.hpp
    submissions/codingame/bots/jacek_nn/replay_value_model.hpp
    submissions/codingame/bots/jacek_nn/teacher_residual_model.hpp
    submissions/codingame/bots/jacek_nn/bot.cpp)
  papersoccer_hash_source_closure(
    PAPERSOCCER_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256
    ${PAPERSOCCER_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_PATHS})

  set(PAPERSOCCER_SELFSEARCH_CONTINUATION_SOURCE_PATHS
    include/papersoccer/bot.hpp
    src/bots/bot.cpp
    src/bots/jacek_replay_bfm/features.cpp
    src/bots/jacek_replay_bfm/model.cpp
    src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp
    src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp
    tools/jacek_replay_continuations.cpp
    tools/jacek_replay_continuations_internal.hpp
    ${PAPERSOCCER_SELFSEARCH_RANK4_ACTOR_SOURCE_PATHS}
    ${PAPERSOCCER_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_PATHS})
  papersoccer_hash_source_closure(
    PAPERSOCCER_SELFSEARCH_CONTINUATION_SOURCE_SHA256
    ${PAPERSOCCER_SELFSEARCH_CONTINUATION_SOURCE_PATHS})

  set(PAPERSOCCER_SELFSEARCH_RANK4_TEACHER_SOURCE_PATHS
    ${PAPERSOCCER_SELFSEARCH_RANK4_ACTOR_SOURCE_PATHS}
    tools/jacek_replay_rank4_position_teacher.cpp)
  papersoccer_hash_source_closure(
    PAPERSOCCER_SELFSEARCH_RANK4_TEACHER_SOURCE_SHA256
    ${PAPERSOCCER_SELFSEARCH_RANK4_TEACHER_SOURCE_PATHS})

  add_executable(papersoccer_jacek_replay_rank4_position_teacher
    tools/jacek_replay_rank4_position_teacher.cpp)
  target_include_directories(papersoccer_jacek_replay_rank4_position_teacher
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_replay_rank4_position_teacher
    PRIVATE papersoccer_core)
  target_compile_definitions(
    papersoccer_jacek_replay_rank4_position_teacher PRIVATE
    PAPERSOCCER_JACEK_REPLAY_RANK4_POSITION_TEACHER_SOURCE_SHA256="${PAPERSOCCER_SELFSEARCH_RANK4_TEACHER_SOURCE_SHA256}")

  set(PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_PATHS
    include/papersoccer/bot.hpp
    include/papersoccer/types.hpp
    include/papersoccer/geometry.hpp
    include/papersoccer/rules.hpp
    src/bots/bot.cpp
    src/bots/mcts_internal.hpp
    src/bots/jacek_replay_bfm/features.cpp
    src/bots/jacek_replay_bfm/model.cpp
    src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp
    src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp
    src/core/geometry.cpp
    src/core/rules.cpp
    tools/jacek_replay_bfm_search_teacher.cpp
    tools/jacek_replay_bfm_search_teacher_internal.hpp)
  set(PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_MATERIAL "")
  foreach(PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_PATH IN LISTS
      PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_PATHS)
    file(SHA256
      "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_PATH}"
      PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_FILE_SHA256)
    string(APPEND PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_MATERIAL
      "${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_PATH}:${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_FILE_SHA256}\n")
  endforeach()
  string(SHA256 PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256
    "${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_MATERIAL}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_PATHS})

  add_executable(papersoccer_jacek_replay_search_teacher
    tools/jacek_replay_bfm_search_teacher.cpp)
  target_include_directories(papersoccer_jacek_replay_search_teacher
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools")
  target_compile_definitions(papersoccer_jacek_replay_search_teacher PRIVATE
    PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256="${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256}")
  target_link_libraries(papersoccer_jacek_replay_search_teacher
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_replay_search_teacher_contract_test
    tests/codingame/jacek_replay_bfm_search_teacher_contract_test.cpp
    tools/jacek_replay_bfm_search_teacher.cpp)
  target_include_directories(
    papersoccer_jacek_replay_search_teacher_contract_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots"
            "${CMAKE_CURRENT_SOURCE_DIR}/tools")
  target_compile_definitions(
    papersoccer_jacek_replay_search_teacher_contract_test PRIVATE
    PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_NO_MAIN=1
    PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256="${PAPERSOCCER_JACEK_REPLAY_SEARCH_TEACHER_SOURCE_SHA256}")
  target_link_libraries(
    papersoccer_jacek_replay_search_teacher_contract_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_replay_search_teacher_contract_test
    COMMAND papersoccer_jacek_replay_search_teacher_contract_test)

  add_executable(papersoccer_jacek_replay_continuations
    tools/jacek_replay_continuations.cpp)
  target_include_directories(papersoccer_jacek_replay_continuations
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_replay_continuations
    PRIVATE papersoccer_core)
  target_compile_definitions(papersoccer_jacek_replay_continuations PRIVATE
    PAPERSOCCER_JACEK_SELFSEARCH_CONTINUATION_SOURCE_SHA256="${PAPERSOCCER_SELFSEARCH_CONTINUATION_SOURCE_SHA256}"
    PAPERSOCCER_JACEK_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256="${PAPERSOCCER_SELFSEARCH_RANK4_ACTOR_SOURCE_SHA256}"
    PAPERSOCCER_JACEK_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256="${PAPERSOCCER_SELFSEARCH_JACEK_NN_ACTOR_SOURCE_SHA256}")
  add_executable(papersoccer_jacek_replay_continuations_contract_test
    tests/codingame/jacek_replay_continuations_contract_test.cpp)
  target_include_directories(
    papersoccer_jacek_replay_continuations_contract_test PRIVATE tools)
  add_test(NAME papersoccer_jacek_replay_continuations_contract_test
    COMMAND papersoccer_jacek_replay_continuations_contract_test)

  add_executable(papersoccer_jacek_replay_comparison
    tools/jacek_replay_bfm_comparison.cpp
    tools/jacek_replay_bfm_rank4_control.cpp
    tools/jacek_replay_bfm_neural_puct_control.cpp
    tools/jacek_replay_bfm_jacek_nn_control.cpp)
  target_include_directories(papersoccer_jacek_replay_comparison
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/rank_4/submission.cpp"
    PAPERSOCCER_JACEK_REPLAY_RANK4_CONTROL_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/rank_4/bot.cpp"
    PAPERSOCCER_JACEK_REPLAY_RANK4_ENGINE_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/neural_puct/submission.cpp"
    PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_CONTROL_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/neural_puct/bot.cpp"
    PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ENGINE_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/jacek_nn/submission.cpp"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_CONTROL_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/jacek_nn/bot.cpp"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ENGINE_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/jacek_replay_bfm_rank4_control.cpp"
    PAPERSOCCER_JACEK_REPLAY_RANK4_ADAPTER_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/jacek_replay_bfm_neural_puct_control.cpp"
    PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ADAPTER_SHA256)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/tools/jacek_replay_bfm_jacek_nn_control.cpp"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ADAPTER_SHA256)
  set(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_PATHS
    include/papersoccer/types.hpp
    include/papersoccer/geometry.hpp
    include/papersoccer/rules.hpp
    src/bots/mcts_internal.hpp
    src/core/geometry.cpp
    src/core/rules.cpp
    submissions/codingame/bots/jacek_nn/replay_book.hpp
    submissions/codingame/bots/jacek_nn/replay_value_model.hpp
    submissions/codingame/bots/jacek_nn/teacher_residual_model.hpp
    submissions/codingame/bots/jacek_nn/bot.cpp
    tools/jacek_replay_bfm_jacek_nn_control.cpp)
  set(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_MATERIAL "")
  foreach(PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_PATH IN LISTS
      PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_PATHS)
    file(SHA256
      "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_PATH}"
      PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_FILE_SHA256)
    string(APPEND PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_MATERIAL
      "${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_PATH}:${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_FILE_SHA256}\n")
  endforeach()
  string(SHA256 PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_SHA256
    "${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_MATERIAL}")
  set(PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_PATHS
    include/papersoccer/types.hpp
    include/papersoccer/geometry.hpp
    include/papersoccer/rules.hpp
    src/core/geometry.cpp
    src/core/rules.cpp
    src/bots/mcts_internal.hpp)
  set(PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_MATERIAL "")
  foreach(PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_PATH IN LISTS
      PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_PATHS)
    file(SHA256
      "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_PATH}"
      PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_FILE_SHA256)
    string(APPEND PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_MATERIAL
      "${PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_PATH}:${PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_FILE_SHA256}\n")
  endforeach()
  string(SHA256 PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_SHA256
    "${PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_MATERIAL}")
  set(PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_PATHS
    include/papersoccer/bot.hpp
    src/bots/bot.cpp
    src/bots/jacek_replay_bfm/features.cpp
    src/bots/jacek_replay_bfm/model.cpp
    src/bots/jacek_replay_bfm/jacek_replay_bfm.cpp
    src/bots/jacek_replay_bfm/jacek_replay_bfm_internal.hpp)
  set(PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_MATERIAL "")
  foreach(PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_PATH IN LISTS
      PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_PATHS)
    file(SHA256
      "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_PATH}"
      PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_FILE_SHA256)
    string(APPEND PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_MATERIAL
      "${PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_PATH}:${PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_FILE_SHA256}\n")
  endforeach()
  string(SHA256 PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_SHA256
    "${PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_MATERIAL}")
  set(PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_PATHS
    tools/jacek_replay_bfm_comparison.cpp
    tools/jacek_replay_bfm_gate_internal.hpp)
  set(PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_MATERIAL "")
  foreach(PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_PATH IN LISTS
      PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_PATHS)
    file(SHA256
      "${CMAKE_CURRENT_SOURCE_DIR}/${PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_PATH}"
      PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_FILE_SHA256)
    string(APPEND PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_MATERIAL
      "${PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_PATH}:${PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_FILE_SHA256}\n")
  endforeach()
  string(SHA256 PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_SHA256
    "${PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_MATERIAL}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    submissions/codingame/bots/rank_4/submission.cpp
    submissions/codingame/bots/rank_4/bot.cpp
    submissions/codingame/bots/neural_puct/submission.cpp
    submissions/codingame/bots/neural_puct/bot.cpp
    submissions/codingame/bots/jacek_nn/submission.cpp
    submissions/codingame/bots/jacek_nn/bot.cpp
    tools/jacek_replay_bfm_rank4_control.cpp
    tools/jacek_replay_bfm_neural_puct_control.cpp
    tools/jacek_replay_bfm_jacek_nn_control.cpp
    ${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_PATHS}
    ${PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_PATHS}
    ${PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_PATHS}
    ${PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_PATHS})
  target_compile_definitions(papersoccer_jacek_replay_comparison PRIVATE
    PAPERSOCCER_JACEK_REPLAY_RANK4_CONTROL_SHA256="${PAPERSOCCER_JACEK_REPLAY_RANK4_CONTROL_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_RANK4_ENGINE_SHA256="${PAPERSOCCER_JACEK_REPLAY_RANK4_ENGINE_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_CONTROL_SHA256="${PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_CONTROL_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ENGINE_SHA256="${PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ENGINE_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_CONTROL_SHA256="${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_CONTROL_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ENGINE_SHA256="${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ENGINE_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_RANK4_ADAPTER_SHA256="${PAPERSOCCER_JACEK_REPLAY_RANK4_ADAPTER_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ADAPTER_SHA256="${PAPERSOCCER_JACEK_REPLAY_NEURAL_PUCT_ADAPTER_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ADAPTER_SHA256="${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_ADAPTER_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_SHA256="${PAPERSOCCER_JACEK_REPLAY_JACEK_NN_SOURCE_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_SHA256="${PAPERSOCCER_JACEK_REPLAY_SHARED_CORE_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_SHA256="${PAPERSOCCER_JACEK_REPLAY_CANDIDATE_SOURCE_SHA256}"
    PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_SHA256="${PAPERSOCCER_JACEK_REPLAY_COMPARISON_SOURCE_SHA256}")
  target_link_libraries(papersoccer_jacek_replay_comparison
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_replay_comparison_smoke
    COMMAND papersoccer_jacek_replay_comparison
      --model
      ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime
      --opponent rank4 --pairs 1 --opening-plies 0 --max-turns 20
      --time-ms 1 --control-work 16 --tree-nodes 32)
  add_test(NAME papersoccer_jacek_replay_comparison_jacek_nn_smoke
    COMMAND papersoccer_jacek_replay_comparison
      --model
      ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime
      --opponent jacek-nn --pairs 1 --opening-plies 0 --max-turns 20
      --time-ms 1 --control-work 16 --tree-nodes 32)
  add_test(NAME papersoccer_jacek_replay_comparison_direct_model_smoke
    COMMAND papersoccer_jacek_replay_comparison
      --model
      ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime
      --control-model
      ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime
      --opponent jacek-replay --pairs 1 --opening-plies 0 --max-turns 20
      --time-ms 1 --tree-nodes 32)
  add_executable(papersoccer_jacek_replay_jacek_nn_control_test
    tests/codingame/jacek_replay_jacek_nn_control_test.cpp
    tools/jacek_replay_bfm_jacek_nn_control.cpp)
  target_include_directories(papersoccer_jacek_replay_jacek_nn_control_test
    PRIVATE tools "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_replay_jacek_nn_control_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_replay_jacek_nn_control_test
    COMMAND papersoccer_jacek_replay_jacek_nn_control_test)
  if(Python3_Interpreter_FOUND)
    add_test(NAME papersoccer_jacek_selfsearch_actor_tools_smoke
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame/jacek_selfsearch_actor_tools_smoke.py
        --continuations $<TARGET_FILE:papersoccer_jacek_replay_continuations>
        --rank4-teacher
        $<TARGET_FILE:papersoccer_jacek_replay_rank4_position_teacher>
        --model
        ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime)
    add_test(NAME papersoccer_jacek_selfsearch_search_teacher_parallel
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame/jacek_selfsearch_search_teacher_parallel_test.py
        --search-teacher
        $<TARGET_FILE:papersoccer_jacek_replay_search_teacher>
        --model
        ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_development/jacek_replay_bfm.runtime)
    add_test(NAME papersoccer_jacek_selfsearch_rank4_teacher_parallel
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame/jacek_selfsearch_rank4_teacher_parallel_test.py
        --rank4-teacher
        $<TARGET_FILE:papersoccer_jacek_replay_rank4_position_teacher>)
    add_test(NAME papersoccer_jacek_replay_comparison_extended_test
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame/test_jacek_replay_comparison_extended.py
        --comparison $<TARGET_FILE:papersoccer_jacek_replay_comparison>
        --model
        ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime)
    add_test(NAME papersoccer_jacek_replay_tool_smoke
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame/jacek_replay_tool_smoke.py
        --repository ${CMAKE_CURRENT_SOURCE_DIR}
        --teacher $<TARGET_FILE:papersoccer_jacek_replay_teacher>
        --continuations $<TARGET_FILE:papersoccer_jacek_replay_continuations>
        --model
        ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_replay_bfm_bootstrap.runtime)
    add_test(NAME papersoccer_jacek_replay_workflow_smoke
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/codingame/jacek_replay_workflow_smoke.py
        --repository ${CMAKE_CURRENT_SOURCE_DIR}
        --teacher $<TARGET_FILE:papersoccer_jacek_replay_teacher>
        --continuations $<TARGET_FILE:papersoccer_jacek_replay_continuations>)
  endif()

  add_executable(papersoccer_jacek_training_data
    tools/jacek_training_data.cpp)
  target_include_directories(papersoccer_jacek_training_data
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_training_data
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_selfplay
    tools/jacek_native_selfplay.cpp)
  target_include_directories(papersoccer_jacek_native_selfplay
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_selfplay
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_native_selfplay_smoke_prepare
    COMMAND ${CMAKE_COMMAND} -E rm -f
      ${CMAKE_CURRENT_BINARY_DIR}/jacek-native-selfplay-smoke.jsonl)
  set_tests_properties(papersoccer_jacek_native_selfplay_smoke_prepare
    PROPERTIES FIXTURES_SETUP papersoccer_jacek_native_selfplay_output)
  add_test(NAME papersoccer_jacek_native_selfplay_smoke
    COMMAND papersoccer_jacek_native_selfplay
      --output
      ${CMAKE_CURRENT_BINARY_DIR}/jacek-native-selfplay-smoke.jsonl
      --games 2 --seed 17 --work 251 --samples-per-game 1
      --temperature 0 --temperature-turns 12 --opening-depths 0,4
      --max-complete-turns 384
      --producer-sha256
      0000000000000000000000000000000000000000000000000000000000000000
      --build-provenance-sha256
      1111111111111111111111111111111111111111111111111111111111111111)
  set_tests_properties(papersoccer_jacek_native_selfplay_smoke
    PROPERTIES FIXTURES_REQUIRED papersoccer_jacek_native_selfplay_output)

  add_executable(papersoccer_jacek_native_selfplay_round2_test
    tests/codingame/jacek_native_selfplay_round2_test.cpp)
  target_include_directories(papersoccer_jacek_native_selfplay_round2_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_selfplay_round2_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_native_selfplay_round2_test
    COMMAND papersoccer_jacek_native_selfplay_round2_test)

  add_executable(papersoccer_jacek_native_selfplay_round2
    tools/jacek_native_selfplay_round2.cpp)
  target_include_directories(papersoccer_jacek_native_selfplay_round2
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_selfplay_round2
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_restart_round2
    tools/jacek_native_restart_round2.cpp)
  target_include_directories(papersoccer_jacek_native_restart_round2
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_restart_round2
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_restart_round2_test
    tests/codingame/jacek_native_restart_round2_test.cpp)
  target_include_directories(papersoccer_jacek_native_restart_round2_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_restart_round2_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_native_restart_round2_test
    COMMAND papersoccer_jacek_native_restart_round2_test)

  add_executable(papersoccer_jacek_native_late_pacing_game_gate
    tools/jacek_native_late_pacing_game_gate.cpp)
  target_include_directories(papersoccer_jacek_native_late_pacing_game_gate
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_late_pacing_game_gate
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_late_pacing_game_gate_test
    tests/codingame/jacek_native_late_pacing_game_gate_test.cpp)
  target_include_directories(
    papersoccer_jacek_native_late_pacing_game_gate_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_late_pacing_game_gate_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_native_late_pacing_game_gate_test
    COMMAND papersoccer_jacek_native_late_pacing_game_gate_test)

  add_executable(papersoccer_jacek_native_replay_decision_auditor
    submissions/codingame/bots/jacek_native_bfm/replay_decision_auditor.cpp)
  target_include_directories(
    papersoccer_jacek_native_replay_decision_auditor
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_jacek_native_replay_decision_auditor
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_replay_decision_auditor_test
    submissions/codingame/bots/jacek_native_bfm/replay_decision_auditor_test.cpp)
  target_include_directories(
    papersoccer_jacek_native_replay_decision_auditor_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_jacek_native_replay_decision_auditor_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_native_replay_decision_auditor_test
    COMMAND papersoccer_jacek_native_replay_decision_auditor_test)

  add_executable(papersoccer_jacek_native_model_gate
    tools/jacek_native_model_gate.cpp)
  target_include_directories(papersoccer_jacek_native_model_gate
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_model_gate
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_search_ab_gate
    tools/jacek_native_search_ab_gate.cpp)
  target_include_directories(papersoccer_jacek_native_search_ab_gate
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_search_ab_gate
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_native_search_ab_gate_test
    tools/jacek_native_search_ab_gate_test.cpp)
  target_include_directories(papersoccer_jacek_native_search_ab_gate_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_native_search_ab_gate_test
    PRIVATE papersoccer_core)
  add_test(NAME papersoccer_jacek_native_search_ab_gate_test
    COMMAND papersoccer_jacek_native_search_ab_gate_test)
  if(Python3_Interpreter_FOUND)
    set(PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT
      "${CMAKE_CURRENT_BINARY_DIR}/jacek-native-current.runtime")
    set(PAPERSOCCER_JACEK_NATIVE_RUNTIME_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/jacek-native-current.hpp")
    set(PAPERSOCCER_JACEK_NATIVE_BASELINE_CHECKPOINT
      "${CMAKE_CURRENT_BINARY_DIR}/jacek-native-seed-20260812.runtime")
    set(PAPERSOCCER_JACEK_NATIVE_BASELINE_HEADER
      "${CMAKE_CURRENT_BINARY_DIR}/jacek-native-seed-20260812.hpp")
    if(PAPERSOCCER_JACEK_NATIVE_ROUND2_DEPLOYMENTS)
      set(PAPERSOCCER_JACEK_NATIVE_GENERATION_ROUTE "round2")
      set(PAPERSOCCER_JACEK_NATIVE_CURRENT_GENERATOR
        ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/jacek_native_round2_activation.py
        generate
        --deployment
        ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_native_round2_deployment.json)
      set(PAPERSOCCER_JACEK_NATIVE_CURRENT_DEPENDS
        models/jacek_native_round2_deployment.json
        submissions/codingame/tools/jacek_native_round2_activation.py
        submissions/codingame/tools/generate_jacek_native_model_round2.py
        submissions/codingame/tools/generate_jacek_native_model.py
        tools/jacek_native_round2_selection.py
        tools/generate_jacek_native_seed.py
        tools/train_jacek_native_round2.py
        tools/train_jacek_native.py
        tools/jacek_native_corpus_round2.py
        tools/jacek_native_restart_corpus_round2.py
        tools/jacek_native_corpus.py
        tools/jacek_native_model_gate.cpp
        submissions/codingame/bots/jacek_native_bfm/bot.cpp
        src/core/rules.cpp
        src/core/geometry.cpp
        src/bots/mcts_internal.hpp
        include/papersoccer/types.hpp
        include/papersoccer/geometry.hpp
        include/papersoccer/rules.hpp
        ${PAPERSOCCER_JACEK_NATIVE_ROUND2_ACTIVATION_INPUTS}
        ${PAPERSOCCER_JACEK_NATIVE_ROUND2_GATE_EVIDENCE})
    else()
      set(PAPERSOCCER_JACEK_NATIVE_GENERATION_ROUTE "round1")
      set(PAPERSOCCER_JACEK_NATIVE_CURRENT_GENERATOR
        ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/generate_jacek_native_model.py
        --model
        ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_native_bootstrap_model.json)
      set(PAPERSOCCER_JACEK_NATIVE_CURRENT_DEPENDS
        models/jacek_native_bootstrap_model.json
        submissions/codingame/tools/generate_jacek_native_model.py
        tools/train_jacek_native.py
        tools/jacek_native_corpus.py)
    endif()
    set(PAPERSOCCER_JACEK_NATIVE_ROUTE_STATE
      "${CMAKE_CURRENT_BINARY_DIR}/jacek-native-generation-route.txt")
    file(GENERATE
      OUTPUT "${PAPERSOCCER_JACEK_NATIVE_ROUTE_STATE}"
      CONTENT "${PAPERSOCCER_JACEK_NATIVE_GENERATION_ROUTE}\n")
    add_custom_command(
      OUTPUT
        ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
        ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_HEADER}
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${PAPERSOCCER_JACEK_NATIVE_CURRENT_GENERATOR}
        --output ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_HEADER}
        --runtime-output ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
      DEPENDS
        ${PAPERSOCCER_JACEK_NATIVE_CURRENT_DEPENDS}
        ${PAPERSOCCER_JACEK_NATIVE_ROUTE_STATE}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
    add_custom_command(
      OUTPUT
        ${PAPERSOCCER_JACEK_NATIVE_BASELINE_CHECKPOINT}
        ${PAPERSOCCER_JACEK_NATIVE_BASELINE_HEADER}
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/generate_jacek_native_model.py
        --model ${CMAKE_CURRENT_SOURCE_DIR}/models/jacek_native_bootstrap_model.json
        --seed 20260812
        --output ${PAPERSOCCER_JACEK_NATIVE_BASELINE_HEADER}
        --runtime-output ${PAPERSOCCER_JACEK_NATIVE_BASELINE_CHECKPOINT}
      DEPENDS
        models/jacek_native_bootstrap_model.json
        submissions/codingame/tools/generate_jacek_native_model.py
        tools/train_jacek_native.py
        tools/jacek_native_corpus.py
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      VERBATIM)
    add_custom_target(papersoccer_jacek_native_runtime_checkpoint
      DEPENDS
        ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
        ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_HEADER}
        ${PAPERSOCCER_JACEK_NATIVE_BASELINE_CHECKPOINT}
        ${PAPERSOCCER_JACEK_NATIVE_BASELINE_HEADER})
    add_dependencies(papersoccer_jacek_native_model_gate
      papersoccer_jacek_native_runtime_checkpoint)
    add_dependencies(papersoccer_jacek_native_search_ab_gate
      papersoccer_jacek_native_runtime_checkpoint)
    add_test(NAME papersoccer_jacek_native_model_gate_checkpoint_smoke
      COMMAND papersoccer_jacek_native_model_gate
        --candidate-checkpoint ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
        --baseline-checkpoint ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
        --verify-only)
    if(NOT PAPERSOCCER_SANITIZERS_ENABLED)
      add_test(NAME papersoccer_jacek_native_model_gate_match_smoke
        COMMAND papersoccer_jacek_native_model_gate
          --candidate-checkpoint ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
          --baseline-checkpoint ${PAPERSOCCER_JACEK_NATIVE_BASELINE_CHECKPOINT}
          --pairs 1 --first-ms 50 --later-ms 10 --opening-turns 0
          --seed 6510615555426900575)
      set_tests_properties(papersoccer_jacek_native_model_gate_match_smoke
        PROPERTIES PASS_REGULAR_EXPRESSION
          "shuffle_seed_policy=deployment-constant")
      add_test(NAME papersoccer_jacek_native_model_gate_varied_shuffle_smoke
        COMMAND papersoccer_jacek_native_model_gate
          --candidate-checkpoint ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
          --baseline-checkpoint ${PAPERSOCCER_JACEK_NATIVE_BASELINE_CHECKPOINT}
          --pairs 1 --first-ms 50 --later-ms 10 --opening-turns 0
          --seed 6510615555426900575 --vary-shuffle-seed)
      set_tests_properties(
        papersoccer_jacek_native_model_gate_varied_shuffle_smoke
        PROPERTIES PASS_REGULAR_EXPRESSION "shuffle_seed_policy=varied")
      add_test(NAME papersoccer_jacek_native_search_ab_gate_match_smoke
        COMMAND papersoccer_jacek_native_search_ab_gate
          --checkpoint
            ${PAPERSOCCER_JACEK_NATIVE_RUNTIME_CHECKPOINT}
          --pairs 1 --first-ms 50 --later-ms 10 --opening-turns 0
          --seed 6510615555426900575
          --candidate-tree-nodes 120000
          --baseline-tree-nodes 80000)
      set_tests_properties(papersoccer_jacek_native_search_ab_gate_match_smoke
        PROPERTIES PASS_REGULAR_EXPRESSION
          "runtime_policy=same.*shuffle_seed_policy=deployment-constant|shuffle_seed_policy=deployment-constant.*runtime_policy=same")
    endif()
  endif()
endif()

find_program(PAPERSOCCER_NODE_EXECUTABLE NAMES node)
if(PAPERSOCCER_NODE_EXECUTABLE)
  execute_process(
    COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --version
    OUTPUT_VARIABLE PAPERSOCCER_NODE_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(PAPERSOCCER_NODE_VERSION MATCHES "^v([0-9]+)" AND
     CMAKE_MATCH_1 GREATER_EQUAL 18)
    add_test(
      NAME papersoccer_web_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_wasm_test.mjs
    )
    add_test(
      NAME papersoccer_web_app_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_app_support_test.mjs
    )
    add_test(
      NAME papersoccer_web_rank5_ui_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_rank5_ui_test.mjs
    )
    add_test(
      NAME papersoccer_web_play_setup_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_play_setup_test.mjs
    )
    add_test(
      NAME papersoccer_web_benchmark_ui_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_benchmark_ui_test.mjs
    )
    add_test(
      NAME papersoccer_web_leaderboard_ui_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_leaderboard_ui_test.mjs
    )
    add_test(
      NAME papersoccer_web_replay_setup_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_replay_setup_test.mjs
    )
    add_test(
      NAME papersoccer_game_review_client_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/game_review_client_test.mjs
    )
    add_test(
      NAME papersoccer_game_review_ui_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/game_review_ui_test.mjs
    )
    add_test(
      NAME papersoccer_web_review_wasm_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/web_review_wasm_test.mjs
    )
    add_test(
      NAME papersoccer_game_review_browser_smoke_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web/game_review_browser_smoke_test.mjs
    )
    set_tests_properties(papersoccer_game_review_browser_smoke_tests PROPERTIES
      TIMEOUT 330
    )
    if(NOT EMSCRIPTEN)
      foreach(PAPERSOCCER_CODINGAME_BOT IN LISTS PAPERSOCCER_CODINGAME_BOTS)
        papersoccer_codingame_target_prefix(
          PAPERSOCCER_CODINGAME_PREFIX "${PAPERSOCCER_CODINGAME_BOT}")
        add_test(
          NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_current
          COMMAND ${PAPERSOCCER_NODE_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/generate_submission.mjs
                  ${PAPERSOCCER_CODINGAME_BOT}
                  --check
        )
        add_test(
          NAME ${PAPERSOCCER_CODINGAME_PREFIX}_protocol_smoke_test
          COMMAND ${PAPERSOCCER_NODE_EXECUTABLE}
                  ${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/tools/protocol_smoke_test.mjs
                  $<TARGET_FILE:${PAPERSOCCER_CODINGAME_PREFIX}_submission>
        )
      endforeach()
      add_test(
        NAME papersoccer_replay_export_tests
        COMMAND ${CMAKE_COMMAND} -E env
                PAPERSOCCER_REPLAY_EXPORTER=$<TARGET_FILE:papersoccer_replay_export>
                ${PAPERSOCCER_NODE_EXECUTABLE} --test
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/replay_export_test.mjs
      )
      add_test(
        NAME papersoccer_arena_cli_tests
        COMMAND ${CMAKE_COMMAND} -E env
                PAPERSOCCER_ARENA=$<TARGET_FILE:papersoccer_arena>
                PAPERSOCCER_OPENING_BANK=$<TARGET_FILE:papersoccer_opening_bank>
                PAPERSOCCER_SANITIZERS_ENABLED=$<BOOL:${PAPERSOCCER_ENABLE_SANITIZERS}>
                ${PAPERSOCCER_NODE_EXECUTABLE} --test
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/arena/arena_cli_test.mjs
      )
    endif()
  endif()
endif()

if(EMSCRIPTEN)
  if(NOT EMSCRIPTEN_VERSION VERSION_EQUAL "6.0.2")
    message(FATAL_ERROR
      "The checked-in browser module is pinned to Emscripten 6.0.2; "
      "found ${EMSCRIPTEN_VERSION}"
    )
  endif()
  add_executable(papersoccer_web_engine src/web/wasm_bridge.cpp)
  target_link_libraries(papersoccer_web_engine PRIVATE papersoccer_core)
  target_compile_options(papersoccer_web_engine PRIVATE
    -O3 -flto -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_web_engine PRIVATE
    --no-entry
    -O3
    -flto
    -sDISABLE_EXCEPTION_CATCHING=0
    -sMODULARIZE=1
    -sEXPORT_NAME=createPaperSoccerModule
    -sEXPORT_ES6=0
    -sSINGLE_FILE=1
    -sSINGLE_FILE_BINARY_ENCODE=0
    -sENVIRONMENT=web,node
    -sFILESYSTEM=0
    -sINITIAL_MEMORY=33554432
    -sALLOW_MEMORY_GROWTH=0
    "-sEXPORTED_RUNTIME_METHODS=['ccall']"
  )
  set_target_properties(papersoccer_web_engine PROPERTIES
    OUTPUT_NAME papersoccer-wasm
    SUFFIX .js
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/web
  )
  add_custom_target(update_papersoccer_web
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:papersoccer_web_engine>
            ${CMAKE_CURRENT_SOURCE_DIR}/web/papersoccer-wasm.js
    DEPENDS papersoccer_web_engine
    COMMENT "Updating the checked-in C++ WebAssembly browser module"
  )
  add_custom_target(check_papersoccer_web
    COMMAND ${CMAKE_COMMAND} -E compare_files
            $<TARGET_FILE:papersoccer_web_engine>
            ${CMAKE_CURRENT_SOURCE_DIR}/web/papersoccer-wasm.js
    DEPENDS papersoccer_web_engine
    COMMENT "Checking that the C++ WebAssembly browser module is current"
  )
  add_test(
    NAME papersoccer_web_artifact_current
    COMMAND ${CMAKE_COMMAND} -E compare_files
            $<TARGET_FILE:papersoccer_web_engine>
            ${CMAKE_CURRENT_SOURCE_DIR}/web/papersoccer-wasm.js
  )

  add_executable(papersoccer_analysis_wasm src/web/review_wasm_bridge.cpp)
  target_link_libraries(papersoccer_analysis_wasm PRIVATE papersoccer_core)
  target_compile_options(papersoccer_analysis_wasm PRIVATE
    -O3 -flto -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_analysis_wasm PRIVATE
    --no-entry
    -O3
    -flto
    -sDISABLE_EXCEPTION_CATCHING=0
    -sMODULARIZE=1
    -sEXPORT_NAME=createPaperSoccerAnalysisModule
    -sEXPORT_ES6=0
    -sSINGLE_FILE=1
    -sSINGLE_FILE_BINARY_ENCODE=0
    -sENVIRONMENT=web,worker,node
    -sFILESYSTEM=0
    -sINITIAL_MEMORY=67108864
    -sALLOW_MEMORY_GROWTH=0
    "-sEXPORTED_RUNTIME_METHODS=['ccall']"
  )
  set_target_properties(papersoccer_analysis_wasm PROPERTIES
    OUTPUT_NAME papersoccer-analysis-wasm
    SUFFIX .js
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/web
  )
  add_custom_target(update_papersoccer_analysis_wasm
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:papersoccer_analysis_wasm>
            ${CMAKE_CURRENT_SOURCE_DIR}/web/papersoccer-analysis-wasm.js
    DEPENDS papersoccer_analysis_wasm
    COMMENT "Updating the checked-in Game Review WebAssembly module"
  )
  add_custom_target(check_papersoccer_analysis_wasm
    COMMAND ${CMAKE_COMMAND} -E compare_files
            $<TARGET_FILE:papersoccer_analysis_wasm>
            ${CMAKE_CURRENT_SOURCE_DIR}/web/papersoccer-analysis-wasm.js
    DEPENDS papersoccer_analysis_wasm
    COMMENT "Checking that the Game Review WebAssembly module is current"
  )
  add_test(
    NAME papersoccer_analysis_wasm_artifact_current
    COMMAND ${CMAKE_COMMAND} -E compare_files
            $<TARGET_FILE:papersoccer_analysis_wasm>
            ${CMAKE_CURRENT_SOURCE_DIR}/web/papersoccer-analysis-wasm.js
  )
endif()
