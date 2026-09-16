# JavaScript, browser, generated-submission, and protocol checks.
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
