add_executable(infersched_smoke_test smoke_test.cpp)
target_link_libraries(infersched_smoke_test
  PRIVATE infersched_core
          infersched_proto
          GTest::gtest_main)

gtest_discover_tests(infersched_smoke_test PROPERTIES LABELS hermetic)

add_executable(infersched_engine_core_test engine_core_test.cpp)
target_link_libraries(infersched_engine_core_test
  PRIVATE infersched_core GTest::gtest_main)

gtest_discover_tests(infersched_engine_core_test PROPERTIES LABELS hermetic)

add_executable(infersched_concurrency_test concurrency_test.cpp)
target_link_libraries(infersched_concurrency_test
  PRIVATE infersched_core GTest::gtest_main)

gtest_discover_tests(infersched_concurrency_test PROPERTIES LABELS hermetic)

add_executable(infersched_durable_test durable_test.cpp)
target_link_libraries(infersched_durable_test
  PRIVATE infersched_durable GTest::gtest_main)
gtest_discover_tests(infersched_durable_test PROPERTIES LABELS live)

add_test(
  NAME Distributed.MultiProcessDispatchAndFencing
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/p4_integration.sh
          $<TARGET_FILE:infersched_router>
          $<TARGET_FILE:infersched_engine>)
set_tests_properties(Distributed.MultiProcessDispatchAndFencing
  PROPERTIES TIMEOUT 20 RUN_SERIAL TRUE LABELS hermetic)

add_test(
  NAME Distributed.EngineCrashRetryWithNewFence
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/p4_crash_recovery.sh
          $<TARGET_FILE:infersched_router>
          $<TARGET_FILE:infersched_engine>)
set_tests_properties(Distributed.EngineCrashRetryWithNewFence
  PROPERTIES TIMEOUT 25 RUN_SERIAL TRUE LABELS hermetic)

add_test(
  NAME Distributed.SameEngineOverlappingRetryRejected
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/p4_same_engine_retry.sh
          $<TARGET_FILE:infersched_router>
          $<TARGET_FILE:infersched_engine>)
set_tests_properties(Distributed.SameEngineOverlappingRetryRejected
  PROPERTIES TIMEOUT 20 RUN_SERIAL TRUE LABELS hermetic)

add_test(
  NAME Durable.KafkaGrpcPostgresEndToEnd
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/p5_end_to_end.sh
          $<TARGET_FILE:infersched_durable_router>
          $<TARGET_FILE:infersched_engine>
          $<TARGET_FILE:infersched_p5_client>)
set_tests_properties(Durable.KafkaGrpcPostgresEndToEnd
  PROPERTIES TIMEOUT 35 RUN_SERIAL TRUE LABELS live)

add_test(
  NAME Durable.ProcessRestartRecoversUnresolvedRequest
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/p5_restart_recovery.sh
          $<TARGET_FILE:infersched_durable_router>
          $<TARGET_FILE:infersched_engine>
          $<TARGET_FILE:infersched_p5_client>)
set_tests_properties(Durable.ProcessRestartRecoversUnresolvedRequest
  PROPERTIES TIMEOUT 35 RUN_SERIAL TRUE LABELS live)

add_test(
  NAME Distributed.MultiRouterKillAndRebalance
  COMMAND ${CMAKE_SOURCE_DIR}/scripts/p6_multi_router_chaos.sh
          $<TARGET_FILE:infersched_durable_router>
          $<TARGET_FILE:infersched_engine>
          $<TARGET_FILE:infersched_p5_client>)
set_tests_properties(Distributed.MultiRouterKillAndRebalance
  PROPERTIES TIMEOUT 55 RUN_SERIAL TRUE LABELS live)

if(EXISTS "${CMAKE_SOURCE_DIR}/.venv/bin/python")
  add_test(
    NAME Python.GatewayUnit
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_SOURCE_DIR}/python
            ${CMAKE_SOURCE_DIR}/.venv/bin/python -m pytest -q
            ${CMAKE_SOURCE_DIR}/python/tests)
  set_tests_properties(Python.GatewayUnit
    PROPERTIES TIMEOUT 20 RUN_SERIAL TRUE LABELS hermetic)

  add_test(
    NAME Distributed.FastApiGatewayEndToEnd
    COMMAND ${CMAKE_SOURCE_DIR}/scripts/p7_gateway_end_to_end.sh
            $<TARGET_FILE:infersched_durable_router>
            $<TARGET_FILE:infersched_engine>
            ${CMAKE_SOURCE_DIR}/.venv/bin/python
            ${CMAKE_SOURCE_DIR})
  set_tests_properties(Distributed.FastApiGatewayEndToEnd
    PROPERTIES TIMEOUT 45 RUN_SERIAL TRUE LABELS live)
endif()
