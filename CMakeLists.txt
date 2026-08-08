cmake_minimum_required(VERSION 3.20)

project(papersoccer LANGUAGES CXX)

# CONFIGURE_DEPENDS notices the validation lock's absent-to-present transition.
# That forces a reconfigure and changes a core compile definition, so an
# incremental post-selection build cannot retain the pre-lock branch.
file(GLOB PAPERSOCCER_GAME_REVIEW_LOCK_FILES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/game_review_gate/game_review_calibration_lock.hpp")
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
endif()

set(PAPERSOCCER_CODINGAME_BOTS
  alpha_beta rank_5 challenger selfplay_nn selfplay_nn_v2 jacek_nn)

function(papersoccer_codingame_target_prefix output_variable bot_name)
  if(bot_name STREQUAL "alpha_beta")
    set(prefix papersoccer_codingame)
  else()
    set(prefix "papersoccer_codingame_${bot_name}")
  endif()
  set(${output_variable} "${prefix}" PARENT_SCOPE)
endfunction()

if(NOT EMSCRIPTEN)
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
    add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test
             COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_submission_test)

    if(EXISTS "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/timing_probe.cpp")
      add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/timing_probe.cpp")
      add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe
               COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_timing_probe)
    endif()

    if(EXISTS "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/comparison_gate.cpp")
      add_executable(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
        "${PAPERSOCCER_CODINGAME_BOT_DIRECTORY}/comparison_gate.cpp")
      target_include_directories(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
      target_link_libraries(${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate
        PRIVATE papersoccer_core)
      add_test(NAME ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate_smoke
        COMMAND ${PAPERSOCCER_CODINGAME_PREFIX}_comparison_gate 1 2000 200)
    endif()
  endforeach()

  add_executable(papersoccer_codingame_selfplay_nn_selfplay_generator
    submissions/codingame/bots/selfplay_nn/selfplay_generator.cpp)
  target_include_directories(
    papersoccer_codingame_selfplay_nn_selfplay_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_selfplay_nn_selfplay_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_selfplay_nn_v2_teacher_sample_generator
    submissions/codingame/bots/selfplay_nn_v2/teacher_sample_generator.cpp)
  target_include_directories(
    papersoccer_codingame_selfplay_nn_v2_teacher_sample_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_selfplay_nn_v2_teacher_sample_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_codingame_jacek_nn_teacher_sample_generator
    submissions/codingame/bots/jacek_nn/teacher_sample_generator.cpp)
  target_include_directories(
    papersoccer_codingame_jacek_nn_teacher_sample_generator
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(
    papersoccer_codingame_jacek_nn_teacher_sample_generator
    PRIVATE papersoccer_core)

  add_executable(papersoccer_jacek_training_data
    tools/jacek_training_data.cpp)
  target_include_directories(papersoccer_jacek_training_data
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bots")
  target_link_libraries(papersoccer_jacek_training_data
    PRIVATE papersoccer_core)
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
