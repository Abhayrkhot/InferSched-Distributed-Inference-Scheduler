# InferSched C++ runtime.
#
# Phase 0 establishes the target graph: a core library that later phases grow
# (Engine core, Router, integrations) plus a test target wired to CTest.

add_library(infersched_core
  src/cost_model.cpp
  src/continuous_engine.cpp
  src/deterministic_engine.cpp
  src/engine_selector.cpp
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
  src/step_executor.cpp
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
  src/request_ingestor.cpp
  src/trace_emitter.cpp)
target_include_directories(infersched_durable PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(infersched_durable
  PUBLIC infersched_proto libpqxx::pqxx PkgConfig::RDKAFKA
  PRIVATE infersched_warnings)

# Redis-backed coordination hints (prefix affinity). Outside the correctness
# boundary: every call degrades to "no hint" when Redis is unavailable.
pkg_check_modules(HIREDIS REQUIRED IMPORTED_TARGET hiredis)
add_library(infersched_coordination src/redis_hints.cpp)
target_include_directories(infersched_coordination PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(infersched_coordination
  PUBLIC PkgConfig::HIREDIS
  PRIVATE infersched_warnings)

# ---------------------------------------------------------------------------
# Optional real-model backend: llama.cpp executes the scheduler's batches.
# Off by default so the control-plane build and CI stay dependency-light.
# ---------------------------------------------------------------------------
option(INFERSCHED_WITH_LLAMA "Build the llama.cpp real-model Engine backend" OFF)
option(INFERSCHED_LLAMA_CUDA "Build llama.cpp with CUDA (requires nvcc)" OFF)
# Also build llama.cpp's own llama-server and llama-batched-bench from the same
# pinned source and the same compiled kernels (the reference baselines).
option(INFERSCHED_LLAMA_TOOLS "Build llama-server and llama-batched-bench too" OFF)
if(INFERSCHED_WITH_LLAMA)
  # Static, like the rest of the project (llama.cpp defaults to shared).
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON ${INFERSCHED_LLAMA_TOOLS} CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS ${INFERSCHED_LLAMA_TOOLS} CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER ${INFERSCHED_LLAMA_TOOLS} CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
  set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
  set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)  # No network at build time; no web UI.
  set(GGML_CUDA ${INFERSCHED_LLAMA_CUDA} CACHE BOOL "" FORCE)
  # Pinned release b12188 (2026-11-07); the same tag builds the reference
  # llama-server used in docs/p10-real-model-serving.md.
  FetchContent_Declare(
    llama
    URL      https://github.com/ggml-org/llama.cpp/archive/refs/tags/b12188.tar.gz
    URL_HASH SHA256=56b07fdceb8641325759a097d43f9f6effc61b2c7c710cb3f61e331e2b1778c6
    SYSTEM)
  FetchContent_MakeAvailable(llama)

  add_library(infersched_llama src/llama_executor.cpp src/gpu_monitor.cpp)
  target_include_directories(infersched_llama PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(infersched_llama
    PUBLIC  infersched_core llama
    PRIVATE infersched_warnings ${CMAKE_DL_LIBS})
  target_compile_definitions(infersched_llama PUBLIC INFERSCHED_HAVE_LLAMA=1)
endif()

add_subdirectory(tests)
add_subdirectory(bench)

add_executable(infersched_engine src/engine_main.cpp)
target_link_libraries(infersched_engine PRIVATE infersched_core)
if(INFERSCHED_WITH_LLAMA)
  target_link_libraries(infersched_engine PRIVATE infersched_llama)
endif()

add_executable(infersched_router src/router_main.cpp)
target_link_libraries(infersched_router PRIVATE infersched_core)

add_executable(infersched_durable_router src/durable_router_main.cpp)
target_link_libraries(infersched_durable_router
  PRIVATE infersched_core infersched_durable infersched_coordination)

add_executable(infersched_p5_client src/p5_client_main.cpp)
target_link_libraries(infersched_p5_client PRIVATE infersched_durable)
