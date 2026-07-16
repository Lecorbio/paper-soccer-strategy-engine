cmake_minimum_required(VERSION 3.20)

project(papersoccer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(papersoccer_core
  src/alpha_beta.cpp
  src/bot.cpp
  src/debug.cpp
  src/geometry.cpp
  src/match.cpp
  src/mcts.cpp
  src/random_bot.cpp
  src/rules.cpp
  src/tactical_search.cpp
  src/web_game.cpp
)
target_include_directories(papersoccer_core PUBLIC include)

add_executable(papersoccer_cli src/cli/main.cpp)
target_link_libraries(papersoccer_cli PRIVATE papersoccer_core)

add_executable(papersoccer_replay_export src/replay/main.cpp)
target_link_libraries(papersoccer_replay_export PRIVATE papersoccer_core)

add_library(papersoccer_arena_support src/arena.cpp)
target_include_directories(papersoccer_arena_support PUBLIC include)
target_link_libraries(papersoccer_arena_support PUBLIC papersoccer_core)

add_executable(papersoccer_arena src/arena/main.cpp)
target_link_libraries(papersoccer_arena PRIVATE papersoccer_arena_support)

add_executable(papersoccer_tests
  tests/alpha_beta_test.cpp
  tests/bot_test.cpp
  tests/match_test.cpp
  tests/mcts_test.cpp
  tests/test_main.cpp
  tests/rules_test.cpp
  tests/web_game_session_test.cpp
)
target_link_libraries(papersoccer_tests PRIVATE papersoccer_core)
target_include_directories(papersoccer_tests PRIVATE src)
if(EMSCRIPTEN)
  target_compile_options(papersoccer_tests PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
  target_link_options(papersoccer_tests PRIVATE -sDISABLE_EXCEPTION_CATCHING=0)
endif()

enable_testing()
add_test(NAME papersoccer_tests COMMAND papersoccer_tests)

if(NOT EMSCRIPTEN)
  add_executable(papersoccer_arena_smoke_test tests/arena_smoke_test.cpp)
  target_link_libraries(papersoccer_arena_smoke_test
    PRIVATE papersoccer_arena_support)
  add_test(NAME papersoccer_arena_smoke_test
           COMMAND papersoccer_arena_smoke_test)
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
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web_wasm_test.mjs
    )
    add_test(
      NAME papersoccer_web_app_tests
      COMMAND ${PAPERSOCCER_NODE_EXECUTABLE} --test
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web_app_support_test.mjs
    )
    if(NOT EMSCRIPTEN)
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
                ${CMAKE_CURRENT_SOURCE_DIR}/tests/arena_cli_test.mjs
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
