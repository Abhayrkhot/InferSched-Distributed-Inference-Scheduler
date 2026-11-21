add_executable(infersched_local_bench scheduler_bench.cpp)
target_link_libraries(infersched_local_bench PRIVATE infersched_core)

# Guards the documented continuous-vs-static batching result (deterministic).
add_test(
  NAME Experiments.ContinuousBatchingOutperformsStaticOnMixedWorkload
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/check_batching_claim.sh
          $<TARGET_FILE:infersched_local_bench>)
set_tests_properties(Experiments.ContinuousBatchingOutperformsStaticOnMixedWorkload
  PROPERTIES TIMEOUT 60 LABELS hermetic)
