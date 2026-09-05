# Optional exact-source candidate checks. Historical harness bytes stay intact.
set(PAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/compact_value_bfm/trained_v2_candidate.cpp"
  CACHE FILEPATH "Exact standalone release candidate to compile and test")

function(papersoccer_add_compact_candidate_checks)
  if(NOT EXISTS "${PAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE}")
    return()
  endif()
  get_filename_component(candidate "${PAPERSOCCER_COMPACT_VALUE_BFM_RELEASE_SOURCE}" ABSOLUTE)
  set(bot "${CMAKE_CURRENT_SOURCE_DIR}/submissions/codingame/bots/compact_value_bfm")
  set(output "${CMAKE_CURRENT_BINARY_DIR}/compact-release-checks")
  file(MAKE_DIRECTORY "${output}")
  add_executable(papersoccer_codingame_compact_value_bfm_release_submission "${candidate}")
  foreach(harness submission_test feature_probe inference_probe)
    set(input "${bot}/${harness}.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${input}")
    file(READ "${input}" contents)
    string(REGEX MATCHALL "#include \"submission.cpp\"" matches "${contents}")
    list(LENGTH matches count)
    if(NOT count EQUAL 1)
      message(FATAL_ERROR "Candidate harness must have exactly one standalone-source include: ${input}")
    endif()
    string(REPLACE "#include \"submission.cpp\"" "#include \"${candidate}\"" replaced "${contents}")
    file(WRITE "${output}/${harness}.cpp" "${replaced}")
    add_executable("papersoccer_codingame_compact_value_bfm_release_${harness}" "${output}/${harness}.cpp")
  endforeach()
  add_test(NAME papersoccer_codingame_compact_value_bfm_release_submission_test
    COMMAND papersoccer_codingame_compact_value_bfm_release_submission_test)
  if(Python3_Interpreter_FOUND)
    add_test(NAME papersoccer_codingame_compact_value_bfm_release_feature_parity
      COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
        ${Python3_EXECUTABLE} "${bot}/feature_parity.py"
        --probe $<TARGET_FILE:papersoccer_codingame_compact_value_bfm_release_feature_probe>
        --states 4096)
    set_tests_properties(papersoccer_codingame_compact_value_bfm_release_feature_parity
      PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" TIMEOUT 120)
  endif()
endfunction()

papersoccer_add_compact_candidate_checks()
