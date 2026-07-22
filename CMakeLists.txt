cmake_minimum_required(VERSION 3.20)

project(papersoccer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(papersoccer_core
  src/bots/alpha_beta.cpp
  src/bots/bot.cpp
  src/bots/mcts.cpp
  src/bots/random_bot.cpp
  src/bots/tactical_search.cpp
  src/core/debug.cpp
  src/core/geometry.cpp
  src/core/match.cpp
  src/core/rules.cpp
  src/web/web_game.cpp
)
target_include_directories(papersoccer_core PUBLIC include)
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

add_executable(papersoccer_arena src/arena/main.cpp)
target_link_libraries(papersoccer_arena PRIVATE papersoccer_arena_support)

if(EMSCRIPTEN)
  target_link_options(papersoccer_cli PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_replay_export PRIVATE
    -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_arena PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
endif()

add_executable(papersoccer_tests
  tests/bots/alpha_beta_test.cpp
  tests/bots/mcts_test.cpp
  tests/bots/random_bot_test.cpp
  tests/core/match_test.cpp
  tests/core/rules_test.cpp
  tests/test_main.cpp
  tests/web/web_game_session_test.cpp
)
target_link_libraries(papersoccer_tests PRIVATE papersoccer_core)
target_include_directories(papersoccer_tests PRIVATE src/bots)
if(EMSCRIPTEN)
  target_compile_options(papersoccer_tests PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_tests PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
endif()

enable_testing()
add_test(NAME papersoccer_tests COMMAND papersoccer_tests)

set(PAPERSOCCER_CODINGAME_BOTS
  alpha_beta rank_5 challenger selfplay_nn selfplay_nn_v2)

function(papersoccer_codingame_target_prefix output_variable bot_name)
  if(bot_name STREQUAL "alpha_beta")
    set(prefix papersoccer_codingame)
  else()
    set(prefix "papersoccer_codingame_${bot_name}")
  endif()
  set(${output_variable} "${prefix}" PARENT_SCOPE)
endfunction()

if(NOT EMSCRIPTEN)
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
  target_compile_options(papersoccer_web_engine PRIVATE -O3 -flto)
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
endif()
