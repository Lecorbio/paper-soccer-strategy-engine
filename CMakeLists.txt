cmake_minimum_required(VERSION 3.20)

project(papersoccer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(papersoccer_core
  src/debug.cpp
  src/geometry.cpp
  src/rules.cpp
)
target_include_directories(papersoccer_core PUBLIC include)

add_executable(papersoccer_cli src/cli/main.cpp)
target_link_libraries(papersoccer_cli PRIVATE papersoccer_core)

add_executable(papersoccer_tests
  tests/test_main.cpp
  tests/rules_test.cpp
)
target_link_libraries(papersoccer_tests PRIVATE papersoccer_core)

enable_testing()
add_test(NAME papersoccer_tests COMMAND papersoccer_tests)
