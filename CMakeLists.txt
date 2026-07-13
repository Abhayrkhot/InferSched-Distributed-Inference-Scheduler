cmake_minimum_required(VERSION 3.20)
project(InferSched LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Standard / defaults
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Sanitizers — ASan and TSan are SEPARATE build configs, never combined in one
# binary (they are mutually exclusive). CI runs them as two independent jobs.
# ---------------------------------------------------------------------------
set(INFERSCHED_SANITIZER "" CACHE STRING "Sanitizer: '' (none), 'address', or 'thread'")
set_property(CACHE INFERSCHED_SANITIZER PROPERTY STRINGS "" "address" "thread")

if(INFERSCHED_SANITIZER STREQUAL "address")
  message(STATUS "InferSched: AddressSanitizer enabled")
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address)
elseif(INFERSCHED_SANITIZER STREQUAL "thread")
  message(STATUS "InferSched: ThreadSanitizer enabled")
  add_compile_options(-fsanitize=thread -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=thread)
elseif(NOT INFERSCHED_SANITIZER STREQUAL "")
  message(FATAL_ERROR "INFERSCHED_SANITIZER must be '', 'address', or 'thread' (got '${INFERSCHED_SANITIZER}')")
endif()

# ---------------------------------------------------------------------------
# Warnings (treat the project's own code strictly; deps are excluded)
# ---------------------------------------------------------------------------
add_library(infersched_warnings INTERFACE)
target_compile_options(infersched_warnings INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf REQUIRED)
endif()
find_package(gRPC CONFIG REQUIRED)
message(STATUS "InferSched: using protobuf ${Protobuf_VERSION}, gRPC ${gRPC_VERSION}")

include(FetchContent)
# GoogleTest — pinned; use a system install if one is already present.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  FIND_PACKAGE_ARGS NAMES GTest
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
include(GoogleTest)

# ---------------------------------------------------------------------------
# Generated gRPC/protobuf contract library (proto/infersched.proto -> C++)
# ---------------------------------------------------------------------------
add_library(infersched_proto STATIC ${CMAKE_CURRENT_SOURCE_DIR}/proto/infersched.proto)
target_link_libraries(infersched_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
target_include_directories(infersched_proto PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")

protobuf_generate(
  TARGET infersched_proto
  LANGUAGE cpp
  APPEND_PATH
  IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/proto"
  PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

protobuf_generate(
  TARGET infersched_proto
  LANGUAGE grpc
  APPEND_PATH
  GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  PLUGIN "protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>"
  IMPORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/proto"
  PROTOC_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------
add_subdirectory(cpp)
