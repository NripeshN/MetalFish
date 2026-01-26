cmake_minimum_required(VERSION 3.20)

# Only enable Objective-C++ on macOS (needed for Metal) Enable CUDA language if
# CUDA support is requested
if(APPLE)
  project(metalfish CXX OBJCXX)
elseif(USE_CUDA)
  project(metalfish CXX CUDA)
else()
  project(metalfish CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
endif()

# Enable NEON for Apple Silicon
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DUSE_NEON=8 -DUSE_NEON_DOTPROD -march=armv8.2-a+dotprod"
  )
  # Disable x86-specific PEXT instruction for ARM
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNO_PEXT")
endif()

# GPU acceleration options
if(APPLE)
  option(USE_METAL "Enable Metal GPU acceleration" ON)
else()
  option(USE_METAL "Enable Metal GPU acceleration" OFF)
endif()

# CUDA support
option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

# Metal-cpp headers location
set(METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp")
set(METAL_CPP_HEADER "${METAL_CPP_DIR}/Metal/Metal.hpp")

# Download metal-cpp if USE_METAL is ON and headers don't exist
if(APPLE AND USE_METAL)
  if(NOT EXISTS "${METAL_CPP_HEADER}")
    message(STATUS "metal-cpp headers not found, downloading...")

    # Create external directory if it doesn't exist
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external")

    # Download metal-cpp from Apple (latest version 26)
    set(METAL_CPP_URL
        "https://developer.apple.com/metal/cpp/files/metal-cpp_26.zip")
    set(METAL_CPP_ZIP "${CMAKE_CURRENT_BINARY_DIR}/metal-cpp.zip")
    set(METAL_CPP_EXTRACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal-cpp-extract")

    # Download
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
      # Extract
      message(STATUS "Extracting metal-cpp...")
      file(ARCHIVE_EXTRACT INPUT ${METAL_CPP_ZIP} DESTINATION
           ${METAL_CPP_EXTRACT_DIR})

      # Move to external/metal-cpp
      file(GLOB METAL_CPP_EXTRACTED_DIR "${METAL_CPP_EXTRACT_DIR}/metal-cpp*")
      if(METAL_CPP_EXTRACTED_DIR)
        # Remove old directory if exists
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

      # Cleanup
      file(REMOVE ${METAL_CPP_ZIP})
      file(REMOVE_RECURSE ${METAL_CPP_EXTRACT_DIR})
    endif()
  endif()

  # Verify headers exist after potential download
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

# ============================================================================
# CUDA Configuration
# ============================================================================
if(USE_CUDA)
  # CUDA optimization options
  option(CUDA_TENSOR_CORES "Enable tensor core kernels (Volta SM 7.0+)" ON)
  option(CUDA_PROFILING "Enable NVTX profiling markers" OFF)
  option(CUDA_WARP_PRIMITIVES "Enable warp-level primitive optimizations" ON)
  
  # Find CUDA toolkit
  find_package(CUDAToolkit QUIET)

  if(CUDAToolkit_FOUND)
    set(CUDA_AVAILABLE ON)
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA include dirs: ${CUDAToolkit_INCLUDE_DIRS}")

    # Check which CUDA targets are available
    set(CUDA_LINK_LIBRARIES "")

    if(TARGET CUDA::cudart_static)
      # Prefer static cudart to avoid runtime dependency issues
      list(APPEND CUDA_LINK_LIBRARIES CUDA::cudart_static)
      message(STATUS "  CUDA::cudart_static: available")
    elseif(TARGET CUDA::cudart)
      list(APPEND CUDA_LINK_LIBRARIES CUDA::cudart)
      message(STATUS "  CUDA::cudart: available")
    else()
      # Try to find cudart manually
      find_library(CUDART_LIBRARY cudart HINTS ${CUDAToolkit_LIBRARY_DIR})
      if(CUDART_LIBRARY)
        list(APPEND CUDA_LINK_LIBRARIES ${CUDART_LIBRARY})
        message(STATUS "  cudart: ${CUDART_LIBRARY}")
      endif()
    endif()

    # Note: We deliberately do NOT link against CUDA::cuda_driver The driver
    # library (libcuda.so) requires an actual NVIDIA GPU with drivers and would
    # prevent running on systems without GPUs (like CI runners) The runtime API
    # (cudart) handles driver loading dynamically
    add_definitions(-DNO_CUDA_DRIVER_API)

    # Note: NVRTC requires the driver API to load compiled PTX, so we disable it
    # too Runtime kernel compilation is not needed - we use pre-compiled kernels
    add_definitions(-DNO_NVRTC)
    message(STATUS "  CUDA Driver API: DISABLED (using Runtime API only)")
    message(STATUS "  NVRTC: DISABLED (pre-compiled kernels only)")

    # Set CUDA architecture (auto-detect or specify)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      # Default to common architectures: Volta, Turing, Ampere, Ada Lovelace,
      # Hopper
      set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()

    # CUDA compiler flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
    
    # Enable tensor core support
    if(CUDA_TENSOR_CORES)
      add_definitions(-DUSE_CUDA_TENSOR_CORES)
      message(STATUS "  Tensor Cores: ENABLED")
    endif()
    
    # Enable NVTX profiling
    if(CUDA_PROFILING)
      add_definitions(-DUSE_NVTX)
      message(STATUS "  NVTX Profiling: ENABLED")
    endif()
    
    # Enable warp primitives
    if(CUDA_WARP_PRIMITIVES)
      add_definitions(-DUSE_CUDA_WARP_PRIMITIVES)
      message(STATUS "  Warp Primitives: ENABLED")
    endif()

    # Enable separable compilation for device code
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  else()
    set(CUDA_AVAILABLE OFF)
    message(WARNING "CUDA Toolkit not found. CUDA support will be disabled.")
    set(USE_CUDA OFF)
  endif()
else()
  set(CUDA_AVAILABLE OFF)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  include_directories(${METAL_CPP_DIR})
endif()
if(USE_CUDA AND CUDA_AVAILABLE)
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
endif()

# Core source files
set(CORE_SOURCES src/core/bitboard.cpp src/core/misc.cpp src/core/movegen.cpp
                 src/core/position.cpp src/core/memory.cpp)

# Search source files
set(SEARCH_SOURCES
    src/search/search.cpp src/search/movepick.cpp src/search/thread.cpp
    src/search/tt.cpp src/search/timeman.cpp)

# Evaluation source files
set(EVAL_SOURCES
    src/eval/evaluate.cpp
    src/eval/score.cpp
    src/eval/nnue/network.cpp
    src/eval/nnue/nnue_accumulator.cpp
    src/eval/nnue/nnue_misc.cpp
    src/eval/nnue/features/full_threats.cpp
    src/eval/nnue/features/half_ka_v2_hm.cpp)

# UCI source files
set(UCI_SOURCES src/uci/uci.cpp src/uci/ucioption.cpp src/uci/engine.cpp
                src/uci/benchmark.cpp)

# Syzygy source files
set(SYZYGY_SOURCES src/syzygy/tbprobe.cpp)

# GPU source files (unified implementation) gpu_nnue_integration.cpp - Main GPU
# NNUE implementation with all optimizations gpu_mcts_backend.cpp - MCTS GPU
# backend gpu_accumulator_cache.cpp - Finny tables and incremental update
# support
set(GPU_SOURCES src/gpu/gpu_nnue_integration.cpp src/gpu/gpu_mcts_backend.cpp
                src/gpu/gpu_accumulator_cache.cpp)

# CUDA source files
set(CUDA_SOURCES "")
if(USE_CUDA AND CUDA_AVAILABLE)
  set(CUDA_SOURCES 
      src/gpu/cuda/cuda_backend.cu
      src/gpu/cuda/cuda_memory.cu
      src/gpu/cuda/kernels/nnue_kernels.cu)
  
  # Add advanced optimization kernels if enabled
  if(CUDA_WARP_PRIMITIVES)
    list(APPEND CUDA_SOURCES src/gpu/cuda/kernels/nnue_simd.cu)
  endif()
  
  if(CUDA_TENSOR_CORES)
    list(APPEND CUDA_SOURCES src/gpu/cuda/kernels/nnue_tensor_core.cu)
  endif()
  
  # Add advanced features
  list(APPEND CUDA_SOURCES 
       src/gpu/cuda/cuda_graphs.cu
       src/gpu/cuda/cuda_multi_gpu.cu
       src/gpu/cuda/cuda_fp16_weights.cu
       src/gpu/cuda/kernels/nnue_persistent.cu)
endif()

# MCTS source files (hybrid search) Core files needed for all MCTS modes: -
# position_adapter: Interface for position representation - position_classifier:
# Classifies positions for strategy selection - ab_integration: Alpha-beta
# integration helpers - thread_safe_mcts: Pure MCTS implementation (stable, used
# by mctsmt and hybrid) - parallel_hybrid_search: Parallel MCTS+AB for
# mcts/hybrid commands - apple_silicon_mcts: Apple Silicon specific
# optimizations
set(MCTS_SOURCES
    src/mcts/position_classifier.cpp
    src/mcts/ab_integration.cpp
    src/mcts/thread_safe_mcts.cpp
    src/mcts/nn_mcts_evaluator.cpp
    src/mcts/position_adapter.cpp
    src/mcts/parallel_hybrid_search.cpp
    src/mcts/apple_silicon_mcts.cpp)

# Find protobuf (minimum version 3.0) - must be before NN_SOURCES
find_package(Protobuf 3.0 REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# Generate protobuf files from .proto definition
# This ensures compatibility with the installed protobuf version
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/proto/net.proto)
set(PROTO_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/proto)
file(MAKE_DIRECTORY ${PROTO_OUTPUT_DIR})

# Custom command to generate protobuf files in the correct location
# The proto file outputs to PROTO_OUTPUT_DIR so includes like "proto/net.pb.h" work
add_custom_command(
    OUTPUT ${PROTO_OUTPUT_DIR}/net.pb.cc ${PROTO_OUTPUT_DIR}/net.pb.h
    COMMAND ${Protobuf_PROTOC_EXECUTABLE}
    ARGS --cpp_out=${PROTO_OUTPUT_DIR}
         --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/src/nn/proto
         ${PROTO_FILE}
    DEPENDS ${PROTO_FILE}
    COMMENT "Generating protobuf files from net.proto"
    VERBATIM
)

# Add generated directory to include path (so "proto/net.pb.h" works from build dir)
# The include path should be the parent of proto/ so that "proto/net.pb.h" resolves
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nn)

# Neural network source files (includes generated protobuf)
set(NN_SOURCES
    ${PROTO_OUTPUT_DIR}/net.pb.cc
    src/nn/loader.cpp
    src/nn/encoder.cpp
    src/nn/policy_map.cpp
    src/nn/network.cpp)

# Metal GPU acceleration (macOS only)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  set(GPU_SOURCES ${GPU_SOURCES} src/gpu/metal/metal_backend.mm)
  set(NN_SOURCES ${NN_SOURCES} src/nn/metal/metal_network.mm)
  
  # Disable ARC for Metal network implementation (uses manual memory management)
  set_source_files_properties(src/nn/metal/metal_network.mm 
                              PROPERTIES COMPILE_FLAGS "-fno-objc-arc")
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_METAL")
  message(STATUS "Metal GPU acceleration: ENABLED")

  # Compile Metal shaders
  set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels)
  set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/metalfish.metallib)

  # Check if Metal compiler is available
  find_program(METAL_COMPILER xcrun)
  if(METAL_COMPILER)
    # Compile single consolidated nnue.metal shader
    add_custom_command(
      OUTPUT ${SHADER_OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E echo "Compiling Metal shaders..."
      COMMAND
        sh -c
        "xcrun -sdk macosx metal -c ${SHADER_DIR}/nnue.metal -o ${CMAKE_CURRENT_BINARY_DIR}/nnue.air 2>&1 || echo 'Metal shader compilation skipped'"
      COMMAND
        sh -c
        "if [ -f ${CMAKE_CURRENT_BINARY_DIR}/nnue.air ]; then xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/nnue.air -o ${SHADER_OUTPUT} 2>&1 || echo 'Metallib creation skipped'; fi"
      COMMAND ${CMAKE_COMMAND} -E touch ${SHADER_OUTPUT}
      DEPENDS ${SHADER_DIR}/nnue.metal
      COMMENT "Compiling Metal shaders"
      VERBATIM)
    add_custom_target(metal_shaders DEPENDS ${SHADER_OUTPUT})
  endif()
elseif(USE_CUDA AND CUDA_AVAILABLE)
  # CUDA GPU acceleration
  add_definitions(-DUSE_CUDA)
  message(STATUS "CUDA GPU acceleration: ENABLED")
  message(STATUS "  CUDA source files: ${CUDA_SOURCES}")
else()
  # CPU fallback backend
  set(GPU_SOURCES ${GPU_SOURCES} src/gpu/cpu_backend.cpp)
  message(STATUS "GPU acceleration: DISABLED (CPU fallback)")
endif()

# All source files
set(SOURCES
    src/main.cpp
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${GPU_SOURCES}
    ${MCTS_SOURCES}
    ${NN_SOURCES})

# Create executable
if(USE_CUDA AND CUDA_AVAILABLE)
  # CUDA executable with mixed source files
  add_executable(metalfish ${SOURCES} ${CUDA_SOURCES})
  set_target_properties(metalfish PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                             CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(metalfish ${CUDA_LINK_LIBRARIES})
else()
  add_executable(metalfish ${SOURCES})
endif()

# Add shader dependency if available
if(TARGET metal_shaders)
  add_dependencies(metalfish metal_shaders)
endif()

# Find zlib (for loading .pb.gz files)
find_package(ZLIB REQUIRED)

# Find absl (required by protobuf >= 22)
find_package(absl CONFIG QUIET)
if(absl_FOUND)
  set(ABSL_LIBS absl::log absl::log_internal_check_op absl::log_internal_message)
else()
  # Try pkg-config as fallback
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(ABSL QUIET absl_log absl_log_internal_check_op)
    if(ABSL_FOUND)
      set(ABSL_LIBS ${ABSL_LIBRARIES})
    endif()
  endif()
endif()

# Link pthread
find_package(Threads REQUIRED)
target_link_libraries(metalfish Threads::Threads ${Protobuf_LIBRARIES} ${ZLIB_LIBRARIES} ${ABSL_LIBS})

# macOS specific
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)

  if(USE_METAL AND METAL_CPP_AVAILABLE)
    target_link_libraries(metalfish ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
                          ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK}
                          ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK}
                          ${ACCELERATE_FRAMEWORK})
  endif()
endif()

# Copy NNUE files to build directory (if they exist)
set(NNUE_FILE1 ${CMAKE_CURRENT_SOURCE_DIR}/src/nn-c288c895ea92.nnue)
set(NNUE_FILE2 ${CMAKE_CURRENT_SOURCE_DIR}/src/nn-37f18f62d772.nnue)

if(EXISTS ${NNUE_FILE1})
  configure_file(${NNUE_FILE1} ${CMAKE_CURRENT_BINARY_DIR}/nn-c288c895ea92.nnue
                 COPYONLY)
  message(STATUS "Found NNUE file: nn-c288c895ea92.nnue")
else()
  message(
    WARNING
      "NNUE file not found: nn-c288c895ea92.nnue - download from https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue"
  )
endif()

if(EXISTS ${NNUE_FILE2})
  configure_file(${NNUE_FILE2} ${CMAKE_CURRENT_BINARY_DIR}/nn-37f18f62d772.nnue
                 COPYONLY)
  message(STATUS "Found NNUE file: nn-37f18f62d772.nnue")
else()
  message(
    WARNING
      "NNUE file not found: nn-37f18f62d772.nnue - download from https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue"
  )
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
      tests/test_gpu_module.cpp
      tests/test_metal.cpp
      tests/test_gpu_nnue.cpp
      tests/test_cuda.cpp)
  
  set(NN_TEST_SOURCES
      tests/test_nn_comparison.cpp
      ${CORE_SOURCES}
      ${NN_SOURCES})

  if(USE_CUDA AND CUDA_AVAILABLE)
    # CUDA test executable
    add_executable(
      metalfish_tests
      ${TEST_SOURCES}
      ${CORE_SOURCES}
      ${SEARCH_SOURCES}
      ${EVAL_SOURCES}
      ${UCI_SOURCES}
      ${SYZYGY_SOURCES}
      ${GPU_SOURCES}
      ${MCTS_SOURCES}
      ${CUDA_SOURCES})
    set_target_properties(
      metalfish_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                 CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(metalfish_tests Threads::Threads
                          ${Protobuf_LIBRARIES} ${ZLIB_LIBRARIES} ${ABSL_LIBS}
                          ${CUDA_LINK_LIBRARIES})
  else()
    add_executable(
      metalfish_tests
      ${TEST_SOURCES}
      ${CORE_SOURCES}
      ${SEARCH_SOURCES}
      ${EVAL_SOURCES}
      ${UCI_SOURCES}
      ${SYZYGY_SOURCES}
      ${GPU_SOURCES}
      ${MCTS_SOURCES}
      ${NN_SOURCES})
    target_link_libraries(metalfish_tests Threads::Threads ${Protobuf_LIBRARIES} ${ZLIB_LIBRARIES} ${ABSL_LIBS})
  endif()

  if(APPLE
     AND USE_METAL
     AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      metalfish_tests ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK}
      ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK}
      ${ACCELERATE_FRAMEWORK})
  endif()

  add_test(NAME metalfish_tests COMMAND metalfish_tests)
  
  # Neural network comparison test
  add_executable(test_nn_comparison ${NN_TEST_SOURCES} ${MCTS_SOURCES})
  target_link_libraries(test_nn_comparison Threads::Threads ${Protobuf_LIBRARIES} ${ZLIB_LIBRARIES} ${ABSL_LIBS})
  
  if(APPLE AND USE_METAL AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      test_nn_comparison ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK}
      ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK})
  endif()
  
  add_test(NAME test_nn_comparison COMMAND test_nn_comparison)
endif()

# ============================================================================
# GPU Benchmark (optional)
# ============================================================================
option(BUILD_GPU_BENCHMARK "Build GPU benchmark" ON)
if(BUILD_GPU_BENCHMARK)
  if(USE_METAL AND METAL_CPP_AVAILABLE)
    add_executable(metalfish_gpu_bench src/benchmark_gpu.cpp ${CORE_SOURCES}
                                       ${GPU_SOURCES})
    target_link_libraries(
      metalfish_gpu_bench Threads::Threads ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK}
      ${QUARTZCORE_FRAMEWORK} ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK})

    # Paper benchmark with full NNUE support
    add_executable(
      metalfish_paper_bench src/paper_benchmark.cpp ${CORE_SOURCES}
                            ${EVAL_SOURCES} ${GPU_SOURCES})
    target_link_libraries(
      metalfish_paper_bench Threads::Threads ${METAL_FRAMEWORK}
      ${FOUNDATION_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK}
      ${QUARTZCORE_FRAMEWORK} ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK})
  elseif(USE_CUDA AND CUDA_AVAILABLE)
    add_executable(metalfish_gpu_bench src/benchmark_gpu.cpp ${CORE_SOURCES}
                                       ${GPU_SOURCES} ${CUDA_SOURCES})
    set_target_properties(
      metalfish_gpu_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                     CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(metalfish_gpu_bench Threads::Threads
                          ${CUDA_LINK_LIBRARIES})

    # Paper benchmark with full NNUE support
    add_executable(
      metalfish_paper_bench src/paper_benchmark.cpp ${CORE_SOURCES}
                            ${EVAL_SOURCES} ${GPU_SOURCES} ${CUDA_SOURCES})
    set_target_properties(
      metalfish_paper_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                       CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(metalfish_paper_bench Threads::Threads
                          ${CUDA_LINK_LIBRARIES})
  endif()
endif()

# ============================================================================
# Print configuration summary
# ============================================================================
message(STATUS "")
message(STATUS "MetalFish Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Metal GPU: ${USE_METAL}")
message(STATUS "  CUDA GPU: ${USE_CUDA}")
if(USE_CUDA AND CUDA_AVAILABLE)
  message(STATUS "    CUDA Version: ${CUDAToolkit_VERSION}")
  message(STATUS "    CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
