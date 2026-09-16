# Python research contracts and frozen artifact checks.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND AND NOT EMSCRIPTEN)
  add_test(
    NAME papersoccer_documentation_links
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_documentation_links.py)
  add_test(
    NAME papersoccer_documentation_link_tests
    COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
            ${Python3_EXECUTABLE} -m unittest tests.test_documentation_links)
  set_tests_properties(papersoccer_documentation_link_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
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
            # Start here deliberately: tests/codingame is not a Python package,
            # so discovery rooted at tests does not recurse into this suite.
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
              tests.codingame.test_jacek_replay_retention
              tests.codingame.test_jacek_rebuild_corpus
              tests.codingame.test_jacek_replay_recovery
              tests.codingame.test_jacek_replay_runtime_v2
              tests.codingame.test_jacek_replay_rebuild
              tests.codingame.test_jacek_replay_rebuild_completion
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

  if(PAPERSOCCER_JACEK_NATIVE_NUMPY_STATUS EQUAL 0)
    add_test(
      NAME papersoccer_compact_value_bfm_pilot_pipeline_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_pilot_pipeline)
    set_tests_properties(
      papersoccer_compact_value_bfm_pilot_pipeline_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
      NAME papersoccer_compact_value_bfm_rank4_teacher_challenger_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_rank4_teacher_challenger)
    set_tests_properties(
      papersoccer_compact_value_bfm_rank4_teacher_challenger_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
      NAME papersoccer_compact_value_bfm_loss_reuse_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_loss_reuse)
    set_tests_properties(
      papersoccer_compact_value_bfm_loss_reuse_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
      NAME papersoccer_compact_value_bfm_rank4_teacher_dual_final_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_rank4_teacher_dual_final)
    set_tests_properties(
      papersoccer_compact_value_bfm_rank4_teacher_dual_final_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
      NAME papersoccer_compact_value_bfm_rank4_teacher_release_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_rank4_teacher_release)
    set_tests_properties(
      papersoccer_compact_value_bfm_rank4_teacher_release_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
      NAME papersoccer_compact_value_bfm_campaign_v2_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              MKL_NUM_THREADS=1
              NUMEXPR_NUM_THREADS=1
              OMP_NUM_THREADS=1
              OPENBLAS_NUM_THREADS=1
              VECLIB_MAXIMUM_THREADS=1
              PAPERSOCCER_COMPACT_TRAINING_THREADS_FIXED_BEFORE_NUMPY=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_campaign_v2
              tests.codingame.test_compact_value_bfm_seed_process_v2
              tests.codingame.test_compact_value_bfm_seed_process_check_v2
              tests.codingame.test_compact_value_bfm_validation_cache_check_v2
              tests.codingame.test_compact_value_bfm_training
              tests.codingame.test_compact_value_bfm_retention_qat
              tests.codingame.test_compact_value_bfm_channel_qat
              tests.codingame.test_compact_value_bfm_channel_integration
              tests.codingame.test_compact_value_bfm_student_rivals
              tests.codingame.test_compact_value_bfm_student_rivals_integration
              tests.codingame.test_compact_value_bfm_prediction_pool
              tests.codingame.test_compact_value_bfm_prediction_integration
              tests.codingame.test_compact_value_bfm_warmup_consistency
              tests.codingame.test_compact_value_bfm_warmup_consistency_integration
              tests.codingame.test_compact_value_bfm_training_outcome
              tests.codingame.test_compact_value_bfm_deterministic_best
              tests.codingame.test_compact_value_bfm_training_resources_v2
              tests.codingame.test_compact_value_bfm_training_acceleration_v2
              tests.codingame.test_compact_value_bfm_training_capacity_v2
              tests.codingame.test_compact_value_bfm_training_workspace_v2
              tests.codingame.test_compact_value_bfm_sparse_scatter
              tests.codingame.test_compact_value_bfm_ranking_store
              tests.codingame.test_compact_value_bfm_parallel_store_v2
              tests.codingame.test_compact_value_bfm_stream_v2
              tests.codingame.test_compact_value_bfm_exclusion_index_v2
              tests.codingame.test_compact_value_bfm_labels_v2
              tests.codingame.test_compact_value_bfm_pilot_selection_v2
              tests.codingame.test_compact_value_bfm_pilot_gate_v2
              tests.codingame.test_compact_value_bfm_pilot_v2
              tests.codingame.test_compact_value_bfm_full_v2
              tests.codingame.test_compact_value_bfm_full_selection_v2
              tests.codingame.test_compact_value_bfm_search_v2
              tests.codingame.test_compact_value_bfm_search_strength_v2
              tests.codingame.test_compact_value_bfm_development_v2
              tests.codingame.test_compact_value_bfm_ci_v2
              tests.codingame.test_compact_value_bfm_timing_instrumentation_v2
              tests.codingame.test_compact_value_bfm_category_profile_v2
              tests.codingame.test_compact_value_bfm_full_outcome_v2
              tests.codingame.test_compact_value_bfm_attribution_v2
              tests.codingame.test_compact_value_bfm_intervention_v2
              tests.codingame.test_compact_value_bfm_terminal_outcome_v2
              tests.codingame.test_compact_value_bfm_release_v2
              tests.codingame.test_compact_value_bfm_protected_v2
              tests.codingame.test_compact_value_bfm_live_v2
              tests.codingame.test_compact_value_bfm_opponent_evaluation
              tests.codingame.test_compact_value_bfm_opponent_suite_v2)
    set_tests_properties(
      papersoccer_compact_value_bfm_campaign_v2_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_test(
      NAME papersoccer_compact_value_bfm_teacher_training_unit_tests
      COMMAND ${CMAKE_COMMAND} -E env
              PYTHONDONTWRITEBYTECODE=1
              ${Python3_EXECUTABLE} -m unittest
              tests.codingame.test_compact_value_bfm_teacher_training)
    set_tests_properties(
      papersoccer_compact_value_bfm_teacher_training_unit_tests PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
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
