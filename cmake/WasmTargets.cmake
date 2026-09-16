# Pinned gameplay and analysis WebAssembly builds.
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
