cmake_minimum_required(VERSION 3.20)

project(papersoccer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(papersoccer_core
  src/bot.cpp
  src/debug.cpp
  src/geometry.cpp
  src/rules.cpp
)
target_include_directories(papersoccer_core PUBLIC include)

add_executable(papersoccer_cli src/cli/main.cpp)
target_link_libraries(papersoccer_cli PRIVATE papersoccer_core)

add_executable(papersoccer_replay_export src/replay/main.cpp)
target_link_libraries(papersoccer_replay_export PRIVATE papersoccer_core)

add_executable(papersoccer_tests
  tests/bot_test.cpp
  tests/test_main.cpp
  tests/rules_test.cpp
)
target_link_libraries(papersoccer_tests PRIVATE papersoccer_core)

enable_testing()
add_test(NAME papersoccer_tests COMMAND papersoccer_tests)

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
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/web_game_test.mjs
    )
  endif()
endif()
