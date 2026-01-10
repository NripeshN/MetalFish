cmake_minimum_required(VERSION 3.20)

# Only enable OBJCXX on macOS (needed for Metal)
if(APPLE)
  project(metalfish CXX OBJCXX)
else()
  project(metalfish CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")

# Enable NEON for Apple Silicon
if(APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} -DUSE_NEON=8 -DUSE_NEON_DOTPROD -march=armv8.2-a+dotprod"
  )
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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  include_directories(${METAL_CPP_DIR})
endif()

# Core source files
set(CORE_SOURCES src/core/bitboard.cpp src/core/misc.cpp src/core/movegen.cpp
                 src/core/position.cpp src/core/memory.cpp)

# Search source files
set(SEARCH_SOURCES
    src/search/search.cpp src/search/movepick.cpp src/search/thread.cpp
    src/search/tt.cpp src/search/timeman.cpp src/search/tune.cpp)

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

# GPU source files (backend-agnostic interface)
set(GPU_SOURCES
    src/gpu/nnue_eval.cpp src/gpu/batch_ops.cpp src/gpu/gpu_nnue.cpp
    src/gpu/gpu_nnue_integration.cpp src/gpu/gpu_accumulator.cpp)

# Metal GPU acceleration (macOS only)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  set(GPU_SOURCES ${GPU_SOURCES} src/gpu/metal/metal_backend.mm)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_METAL")
  message(STATUS "Metal GPU acceleration: ENABLED")

  # Compile Metal shaders (optional - shaders can also be compiled at runtime)
  set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/metal/kernels)
  set(SHADER_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/metalfish.metallib)

  # Check if Metal compiler is available
  find_program(METAL_COMPILER xcrun)
  if(METAL_COMPILER)
    # Create a shell script to compile shaders with proper error handling
    add_custom_command(
      OUTPUT ${SHADER_OUTPUT}
      COMMAND ${CMAKE_COMMAND} -E echo "Compiling Metal shaders..."
      COMMAND
        sh -c
        "xcrun -sdk macosx metal -c ${SHADER_DIR}/nnue_full.metal -o ${CMAKE_CURRENT_BINARY_DIR}/nnue.air 2>&1 || echo 'Metal shader compilation skipped'"
      COMMAND
        sh -c
        "if [ -f ${CMAKE_CURRENT_BINARY_DIR}/nnue.air ]; then xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/nnue.air -o ${SHADER_OUTPUT} 2>&1 || echo 'Metallib creation skipped'; fi"
      COMMAND ${CMAKE_COMMAND} -E touch ${SHADER_OUTPUT}
      DEPENDS ${SHADER_DIR}/nnue_full.metal
      COMMENT "Compiling Metal shaders (optional)"
      VERBATIM)
    add_custom_target(metal_shaders DEPENDS ${SHADER_OUTPUT})
  endif()
else()
  # CPU fallback backend
  set(GPU_SOURCES ${GPU_SOURCES} src/gpu/cpu_backend.cpp)
  message(STATUS "Metal GPU acceleration: DISABLED (CPU fallback)")
endif()

# CUDA GPU acceleration
if(USE_CUDA)
  # Check if CUDA is available
  include(CheckLanguage)
  check_language(CUDA)
  
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    
    # Set CUDA standard
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # Add CUDA backend source
    set(GPU_SOURCES ${GPU_SOURCES} src/gpu/cuda/cuda_backend.cu)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_CUDA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DUSE_CUDA")
    
    # Set CUDA architectures (supporting common GPUs)
    # Pascal (6.0, 6.1), Volta (7.0), Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;75;80;86;89;90")
    
    message(STATUS "CUDA GPU acceleration: ENABLED")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message(WARNING "CUDA compiler not found. CUDA support will be disabled.")
    set(USE_CUDA OFF)
  endif()
elseif(NOT USE_METAL OR NOT METAL_CPP_AVAILABLE)
  # Only add CPU backend if neither Metal nor CUDA is enabled
  # and it hasn't been added already
  if(NOT USE_METAL OR NOT METAL_CPP_AVAILABLE)
    # Check if cpu_backend.cpp is not already in GPU_SOURCES
    list(FIND GPU_SOURCES "src/gpu/cpu_backend.cpp" _index)
    if(_index EQUAL -1)
      set(GPU_SOURCES ${GPU_SOURCES} src/gpu/cpu_backend.cpp)
    endif()
  endif()
endif()

# All source files
set(SOURCES
    src/main.cpp
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${GPU_SOURCES})

# Create executable
add_executable(metalfish ${SOURCES})

# Add shader dependency if available
if(TARGET metal_shaders)
  add_dependencies(metalfish metal_shaders)
endif()

# Link pthread
find_package(Threads REQUIRED)
target_link_libraries(metalfish Threads::Threads)

# macOS specific
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore)

  if(USE_METAL AND METAL_CPP_AVAILABLE)
    target_link_libraries(metalfish ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
                          ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
  endif()
endif()

# CUDA specific
if(USE_CUDA AND CMAKE_CUDA_COMPILER)
  # Find CUDA libraries
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(metalfish CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
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

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()

  set(TEST_SOURCES
      tests/test_main.cpp
      tests/test_bitboard.cpp
      tests/test_position.cpp
      tests/test_movegen.cpp
      tests/test_search.cpp
      tests/test_metal.cpp
      tests/test_cuda.cpp
      tests/test_gpu_nnue.cpp)

  add_executable(
    metalfish_tests
    ${TEST_SOURCES}
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${GPU_SOURCES})
  target_link_libraries(metalfish_tests Threads::Threads)

  if(APPLE
     AND USE_METAL
     AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      metalfish_tests ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
  endif()
  
  if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(metalfish_tests CUDA::cudart CUDA::cuda_driver CUDA::nvrtc)
  endif()

  add_test(NAME metalfish_tests COMMAND metalfish_tests)
endif()

# GPU Benchmark (optional)
option(BUILD_GPU_BENCHMARK "Build GPU benchmark" OFF)
if(BUILD_GPU_BENCHMARK
   AND USE_METAL
   AND METAL_CPP_AVAILABLE)
  add_executable(metalfish_gpu_bench src/benchmark_gpu.cpp ${CORE_SOURCES}
                                     ${GPU_SOURCES})
  target_link_libraries(
    metalfish_gpu_bench Threads::Threads ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK} ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "MetalFish Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Metal GPU: ${USE_METAL}")
message(STATUS "  CUDA GPU: ${USE_CUDA}")
message(STATUS "  Tests: ${BUILD_TESTS}")
message(STATUS "")
