cmake_minimum_required(VERSION 3.20)

# MetalFish - A GPU-accelerated UCI chess engine for Apple Silicon
if(APPLE)
  project(metalfish CXX OBJCXX)
else()
  project(metalfish CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  add_compile_definitions(IS_64BIT)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type" FORCE)
endif()

option(METALFISH_ENABLE_IPO
       "Enable interprocedural optimization for Release builds" ON)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  option(METALFISH_ENABLE_NATIVE_ARCH
         "Tune code generation for the local Apple Silicon CPU" ON)
else()
  option(METALFISH_ENABLE_NATIVE_ARCH "Tune code generation for the local CPU"
         OFF)
endif()

set(METALFISH_IPO_SUPPORTED OFF)
if(METALFISH_ENABLE_IPO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT METALFISH_IPO_SUPPORTED OUTPUT METALFISH_IPO_ERROR)
  if(NOT METALFISH_IPO_SUPPORTED)
    message(STATUS "IPO/LTO disabled: ${METALFISH_IPO_ERROR}")
  endif()
endif()

# Enable ARM-specific optimizations for Apple Silicon.
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(APPLE_ARM_FLAGS "-DUSE_NEON=8 -DUSE_NEON_DOTPROD -DUSE_POPCNT -DNO_PEXT")
  if(METALFISH_ENABLE_NATIVE_ARCH)
    string(APPEND APPLE_ARM_FLAGS " -mcpu=native -mtune=native")
  else()
    string(APPEND APPLE_ARM_FLAGS " -mcpu=apple-m1 -mtune=apple-m1")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${APPLE_ARM_FLAGS}")
  set(CMAKE_OBJCXX_FLAGS "${CMAKE_OBJCXX_FLAGS} ${APPLE_ARM_FLAGS}")
endif()

# Metal transformer acceleration (Apple Silicon)
if(APPLE)
  option(USE_METAL "Enable Metal transformer acceleration" ON)
else()
  option(USE_METAL "Enable Metal transformer acceleration" OFF)
endif()
option(USE_CUDA "Enable CUDA transformer backend entry point" OFF)
set(METALFISH_CUDA_LIBS "")

# Metal-cpp headers
set(METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp")
set(METAL_CPP_HEADER "${METAL_CPP_DIR}/Metal/Metal.hpp")

if(APPLE AND USE_METAL)
  if(NOT EXISTS "${METAL_CPP_HEADER}")
    message(STATUS "metal-cpp headers not found, downloading...")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external")
    set(METAL_CPP_URL
        "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip")
    set(METAL_CPP_ZIP "${CMAKE_CURRENT_BINARY_DIR}/metal-cpp.zip")
    set(METAL_CPP_EXTRACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal-cpp-extract")
    file(
      DOWNLOAD ${METAL_CPP_URL} ${METAL_CPP_ZIP}
      STATUS DOWNLOAD_STATUS
      SHOW_PROGRESS)
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    if(NOT DOWNLOAD_RESULT EQUAL 0)
      message(
        WARNING "Failed to download metal-cpp. Metal support will be disabled.")
      set(USE_METAL OFF)
    else()
      message(STATUS "Extracting metal-cpp...")
      file(ARCHIVE_EXTRACT INPUT ${METAL_CPP_ZIP} DESTINATION
           ${METAL_CPP_EXTRACT_DIR})
      file(GLOB METAL_CPP_EXTRACTED_DIR "${METAL_CPP_EXTRACT_DIR}/metal-cpp*")
      if(METAL_CPP_EXTRACTED_DIR)
        if(EXISTS "${METAL_CPP_DIR}")
          file(REMOVE_RECURSE "${METAL_CPP_DIR}")
        endif()
        file(RENAME "${METAL_CPP_EXTRACTED_DIR}" "${METAL_CPP_DIR}")
        message(STATUS "metal-cpp installed to ${METAL_CPP_DIR}")
      else()
        message(
          WARNING "Failed to extract metal-cpp. Metal support will be disabled."
        )
        set(USE_METAL OFF)
      endif()
      file(REMOVE ${METAL_CPP_ZIP})
      file(REMOVE_RECURSE ${METAL_CPP_EXTRACT_DIR})
    endif()
  endif()

  if(EXISTS "${METAL_CPP_HEADER}")
    set(METAL_CPP_AVAILABLE ON)
  else()
    set(METAL_CPP_AVAILABLE OFF)
    message(
      WARNING
        "metal-cpp headers still not available. Metal support will be disabled."
    )
    set(USE_METAL OFF)
  endif()
else()
  set(METAL_CPP_AVAILABLE OFF)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  include_directories(${METAL_CPP_DIR})
endif()

# ============================================================================
# Source files organized by module
# ============================================================================

# Core chess primitives
set(CORE_SOURCES src/core/bitboard.cpp src/core/misc.cpp src/core/movegen.cpp
                 src/core/position.cpp src/core/memory.cpp)

# Alpha-Beta search engine
set(SEARCH_SOURCES
    src/search/search.cpp src/search/movepick.cpp src/search/thread.cpp
    src/search/tt.cpp src/search/timeman.cpp)

# NNUE evaluation. NNUE stays CPU-only; Metal is reserved for transformer MCTS.
set(EVAL_SOURCES
    src/eval/evaluate.cpp
    src/eval/score.cpp
    src/eval/nnue/network.cpp
    src/eval/nnue/nnue_accumulator.cpp
    src/eval/nnue/nnue_misc.cpp
    src/eval/nnue/features/full_threats.cpp
    src/eval/nnue/features/half_ka_v2_hm.cpp)

# UCI protocol
set(UCI_SOURCES src/uci/uci.cpp src/uci/ucioption.cpp src/uci/engine.cpp
                src/uci/benchmark.cpp)

# Tablebase probing
set(SYZYGY_SOURCES src/syzygy/tbprobe.cpp)

# MCTS search engine
set(MCTS_SOURCES src/mcts/search.cpp src/mcts/backend_adapter.cpp
                 src/mcts/stoppers.cpp src/mcts/evaluator.cpp)

# Hybrid search engine
set(HYBRID_SOURCES src/hybrid/hybrid_search.cpp src/hybrid/position_adapter.cpp
                   src/hybrid/classifier.cpp)

# Protobuf generation for transformer network weights
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

if(TARGET protobuf::libprotobuf)
  set(METALFISH_PROTOBUF_LIB protobuf::libprotobuf)
elseif(TARGET libprotobuf)
  set(METALFISH_PROTOBUF_LIB libprotobuf)
else()
  set(METALFISH_PROTOBUF_LIB ${Protobuf_LIBRARIES})
endif()

set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/proto/net.proto)
set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

add_custom_command(
  OUTPUT ${PROTO_OUTPUT_DIR}/net.pb.cc ${PROTO_OUTPUT_DIR}/net.pb.h
  COMMAND ${Protobuf_PROTOC_EXECUTABLE} ARGS --cpp_out=${PROTO_OUTPUT_DIR}
          --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/nn/proto ${PROTO_FILE}
  DEPENDS ${PROTO_FILE} ${Protobuf_PROTOC_EXECUTABLE}
  COMMENT "Generating protobuf files from net.proto"
  VERBATIM)

if(APPLE AND METALFISH_IPO_SUPPORTED)
  set_source_files_properties(${PROTO_OUTPUT_DIR}/net.pb.cc
                              PROPERTIES COMPILE_FLAGS "-fno-lto")
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nn)

# Transformer neural network (for MCTS)
set(NN_SOURCES
    ${PROTO_OUTPUT_DIR}/net.pb.cc
    src/nn/loader.cpp
    src/nn/encoder.cpp
    src/nn/weights.cpp
    src/nn/policy_map.cpp
    src/nn/network_format.cpp
    src/nn/network_tensor_plan.cpp
    src/nn/network_weight_inventory.cpp
    src/nn/network_output_decoder.cpp
    src/nn/network.cpp)

# Metal transformer acceleration (macOS only)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  # Generic Metal device/buffer backend used by diagnostics and shutdown. NNUE
  # is deliberately not compiled for Metal; CPU incremental NNUE is faster for
  # AB search and avoids starving transformer inference on Apple GPUs.
  set(EVAL_SOURCES ${EVAL_SOURCES} src/eval/metal/metal_backend.mm)

  # Transformer Metal/MPSGraph backend
  set(NN_SOURCES
      ${NN_SOURCES} src/nn/metal/metal_network.mm
      src/nn/metal/mps/MetalNetworkBuilder.mm src/nn/metal/mps/NetworkGraph.mm)

  # Enable ARC for Metal ObjC++ files (required for MPSGraph)
  set_source_files_properties(
    src/nn/metal/metal_network.mm src/nn/metal/mps/MetalNetworkBuilder.mm
    src/nn/metal/mps/NetworkGraph.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_METAL")
  message(STATUS "Metal transformer acceleration: ENABLED")
else()
  # CPU fallback backend
  set(EVAL_SOURCES ${EVAL_SOURCES} src/eval/cpu_backend.cpp)
  message(STATUS "GPU acceleration: DISABLED (CPU fallback)")
endif()

if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
        75 80 86 89
        CACHE STRING "CUDA architectures for NVIDIA tournament hosts")
  endif()
  find_package(CUDAToolkit REQUIRED)
  set(NN_SOURCES
      ${NN_SOURCES}
      src/nn/cuda/cuda_executor.cpp
      src/nn/cuda/cuda_network.cpp
      src/nn/cuda/cuda_null_executor_smoke.cpp
      src/nn/cuda/cuda_runtime_probe.cu
      src/nn/cuda/cuda_input_packing.cu
      src/nn/cuda/cuda_buffers.cu
      src/nn/cuda/cuda_weight_buffers.cu)
  set(METALFISH_CUDA_LIBS CUDA::cudart)
  add_compile_definitions(USE_CUDA)
  message(STATUS "CUDA transformer backend: ENABLED (entry point)")
endif()

# ============================================================================
# Build targets
# ============================================================================

# All source files
set(SOURCES
    src/main.cpp
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${MCTS_SOURCES}
    ${HYBRID_SOURCES}
    ${NN_SOURCES})

add_executable(metalfish ${SOURCES})

if(METALFISH_IPO_SUPPORTED)
  set_property(TARGET metalfish PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE
                                         TRUE)
endif()

# Libraries
find_package(ZLIB REQUIRED)
if(TARGET ZLIB::ZLIB)
  set(METALFISH_ZLIB_LIB ZLIB::ZLIB)
else()
  set(METALFISH_ZLIB_LIB ${ZLIB_LIBRARIES})
endif()

set(ABSL_LIBS "")
find_package(absl CONFIG QUIET)
if(absl_FOUND)
  foreach(ABSL_TARGET absl::log absl::log_internal_check_op
                      absl::log_internal_message)
    if(TARGET ${ABSL_TARGET})
      list(APPEND ABSL_LIBS ${ABSL_TARGET})
    endif()
  endforeach()
  if(ABSL_LIBS)
    message(STATUS "Found abseil CMake targets: ${ABSL_LIBS}")
  endif()
else()
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(ABSL_PKG QUIET absl_log)
    if(ABSL_PKG_FOUND)
      message(STATUS "Found abseil via pkg-config")
      set(ABSL_LIBS ${ABSL_PKG_LIBRARIES})
      include_directories(${ABSL_PKG_INCLUDE_DIRS})
      link_directories(${ABSL_PKG_LIBRARY_DIRS})
    endif()
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(metalfish Threads::Threads ${METALFISH_PROTOBUF_LIB}
                      ${METALFISH_ZLIB_LIB} ${ABSL_LIBS} ${METALFISH_CUDA_LIBS})

# macOS frameworks
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)

  if(ACCELERATE_FRAMEWORK)
    target_link_libraries(metalfish ${ACCELERATE_FRAMEWORK})
  endif()

  if(USE_METAL AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      metalfish
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK}
      ${QUARTZCORE_FRAMEWORK}
      ${MPS_FRAMEWORK}
      ${MPSGRAPH_FRAMEWORK})
  endif()
endif()

# Copy NNUE files to build directory
set(NNUE_FILE1 ${CMAKE_CURRENT_SOURCE_DIR}/networks/nn-c288c895ea92.nnue)
set(NNUE_FILE2 ${CMAKE_CURRENT_SOURCE_DIR}/networks/nn-37f18f62d772.nnue)

if(EXISTS ${NNUE_FILE1})
  configure_file(${NNUE_FILE1} ${CMAKE_CURRENT_BINARY_DIR}/nn-c288c895ea92.nnue
                 COPYONLY)
  message(STATUS "Found NNUE file: nn-c288c895ea92.nnue")
endif()
if(EXISTS ${NNUE_FILE2})
  configure_file(${NNUE_FILE2} ${CMAKE_CURRENT_BINARY_DIR}/nn-37f18f62d772.nnue
                 COPYONLY)
  message(STATUS "Found NNUE file: nn-37f18f62d772.nnue")
endif()

# ============================================================================
# Tests
# ============================================================================
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()

  set(TEST_SOURCES
      tests/test_main.cpp
      tests/test_core.cpp
      tests/test_search_module.cpp
      tests/test_mcts_module.cpp
      tests/test_hybrid.cpp
      tests/test_eval_gpu.cpp
      tests/nn_input_fixture.cpp)

  add_executable(
    metalfish_tests
    ${TEST_SOURCES}
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${MCTS_SOURCES}
    ${HYBRID_SOURCES}
    ${NN_SOURCES})
  if(METALFISH_IPO_SUPPORTED)
    set_property(TARGET metalfish_tests
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
  target_link_libraries(
    metalfish_tests Threads::Threads ${METALFISH_PROTOBUF_LIB}
    ${METALFISH_ZLIB_LIB} ${ABSL_LIBS} ${METALFISH_CUDA_LIBS})

  if(APPLE AND ACCELERATE_FRAMEWORK)
    target_link_libraries(metalfish_tests ${ACCELERATE_FRAMEWORK})
  endif()

  if(APPLE
     AND USE_METAL
     AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      metalfish_tests
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK}
      ${QUARTZCORE_FRAMEWORK}
      ${MPS_FRAMEWORK}
      ${MPSGRAPH_FRAMEWORK})
  endif()
  add_test(NAME metalfish_tests COMMAND metalfish_tests)

  # Neural network comparison test
  add_executable(
    test_nn_comparison
    tests/test_nn_comparison.cpp
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${MCTS_SOURCES}
    ${HYBRID_SOURCES}
    ${NN_SOURCES})
  if(METALFISH_IPO_SUPPORTED)
    set_property(TARGET test_nn_comparison
                 PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
  target_link_libraries(
    test_nn_comparison Threads::Threads ${METALFISH_PROTOBUF_LIB}
    ${METALFISH_ZLIB_LIB} ${ABSL_LIBS} ${METALFISH_CUDA_LIBS})

  if(APPLE AND ACCELERATE_FRAMEWORK)
    target_link_libraries(test_nn_comparison ${ACCELERATE_FRAMEWORK})
  endif()

  if(APPLE
     AND USE_METAL
     AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      test_nn_comparison
      ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK}
      ${QUARTZCORE_FRAMEWORK}
      ${MPS_FRAMEWORK}
      ${MPSGRAPH_FRAMEWORK})
  endif()
  add_test(NAME test_nn_comparison COMMAND test_nn_comparison)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "MetalFish Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Metal transformer: ${USE_METAL}")
message(STATUS "  Native arch tuning: ${METALFISH_ENABLE_NATIVE_ARCH}")
message(STATUS "  IPO/LTO: ${METALFISH_IPO_SUPPORTED}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
