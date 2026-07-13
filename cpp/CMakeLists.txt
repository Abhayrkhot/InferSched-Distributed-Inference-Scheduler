# InferSched C++ runtime.
#
# Phase 0 establishes the target graph: a core library that later phases grow
# (Engine core, Router, integrations) plus a test target wired to CTest.

add_library(infersched_core
  src/cost_model.cpp
  src/continuous_engine.cpp
  src/deterministic_engine.cpp
  src/execution_state.cpp
  src/engine_service.cpp
  src/histogram.cpp
  src/paged_kv_cache.cpp
  src/prefix_cache.cpp
  src/radix_prefix_cache.cpp
  src/request_queue.cpp
  src/router_service.cpp
  src/result_cache.cpp
  src/result_cache_key.cpp
  src/static_batch_policy.cpp
  src/version.cpp)
target_include_directories(infersched_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(infersched_core
  PUBLIC  infersched_proto      # the generated gRPC/protobuf contract
  PRIVATE infersched_warnings)

find_package(PkgConfig REQUIRED)
find_package(libpqxx 8 CONFIG REQUIRED)
pkg_check_modules(RDKAFKA REQUIRED IMPORTED_TARGET rdkafka++)
add_library(infersched_durable
  src/durable_store.cpp
  src/kafka_poller.cpp
  src/offset_tracker.cpp
  src/outbox_relay.cpp
  src/request_ingestor.cpp)
target_include_directories(infersched_durable PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(infersched_durable
  PUBLIC infersched_proto libpqxx::pqxx PkgConfig::RDKAFKA
  PRIVATE infersched_warnings)

add_subdirectory(tests)
add_subdirectory(bench)

add_executable(infersched_engine src/engine_main.cpp)
target_link_libraries(infersched_engine PRIVATE infersched_core)

add_executable(infersched_router src/router_main.cpp)
target_link_libraries(infersched_router PRIVATE infersched_core)

add_executable(infersched_durable_router src/durable_router_main.cpp)
target_link_libraries(infersched_durable_router
  PRIVATE infersched_core infersched_durable)

add_executable(infersched_p5_client src/p5_client_main.cpp)
target_link_libraries(infersched_p5_client PRIVATE infersched_durable)
