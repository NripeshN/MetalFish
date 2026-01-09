cmake_minimum_required(VERSION 3.20)
project(metalfish CXX OBJCXX)

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

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp)

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

# Metal GPU acceleration source files (macOS only)
set(METAL_SOURCES)
if(APPLE)
  set(METAL_SOURCES src/metal/metal_device.mm src/metal/device.cpp
                    src/metal/allocator.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_METAL")
endif()

# All source files
set(SOURCES
    src/main.cpp
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${METAL_SOURCES})

# Create executable
add_executable(metalfish ${SOURCES})

# Link pthread
find_package(Threads REQUIRED)
target_link_libraries(metalfish Threads::Threads)

# macOS specific
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
  find_library(QUARTZCORE_FRAMEWORK QuartzCore)

  target_link_libraries(metalfish ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
                        ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})

  # Compile Metal shaders (disabled - requires Metal toolchain) set(SHADER_DIR
  # ${CMAKE_CURRENT_SOURCE_DIR}/shaders) set(SHADER_OUTPUT
  # ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
  #
  # add_custom_command( OUTPUT ${SHADER_OUTPUT} COMMAND xcrun -sdk macosx metal
  # -c ${SHADER_DIR}/core_ops.metal -o ${CMAKE_CURRENT_BINARY_DIR}/core_ops.air
  # 2>/dev/null || true COMMAND xcrun -sdk macosx metal -c
  # ${SHADER_DIR}/nnue.metal -o ${CMAKE_CURRENT_BINARY_DIR}/nnue.air 2>/dev/null
  # || true COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/*.air
  # -o ${SHADER_OUTPUT} 2>/dev/null || true DEPENDS ${SHADER_DIR}/core_ops.metal
  # ${SHADER_DIR}/nnue.metal COMMENT "Compiling Metal shaders" VERBATIM )
  # add_custom_target(metal_shaders DEPENDS ${SHADER_OUTPUT})
  # add_dependencies(metalfish metal_shaders)
endif()

# Copy NNUE files to build directory (if they exist)
set(NNUE_FILE1 ${CMAKE_CURRENT_SOURCE_DIR}/src/nn-c288c895ea92.nnue)
set(NNUE_FILE2 ${CMAKE_CURRENT_SOURCE_DIR}/src/nn-37f18f62d772.nnue)

if(EXISTS ${NNUE_FILE1})
  configure_file(${NNUE_FILE1} ${CMAKE_CURRENT_BINARY_DIR}/nn-c288c895ea92.nnue COPYONLY)
  message(STATUS "Found NNUE file: nn-c288c895ea92.nnue")
else()
  message(WARNING "NNUE file not found: nn-c288c895ea92.nnue - download from https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue")
endif()

if(EXISTS ${NNUE_FILE2})
  configure_file(${NNUE_FILE2} ${CMAKE_CURRENT_BINARY_DIR}/nn-37f18f62d772.nnue COPYONLY)
  message(STATUS "Found NNUE file: nn-37f18f62d772.nnue")
else()
  message(WARNING "NNUE file not found: nn-37f18f62d772.nnue - download from https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue")
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
  enable_testing()

  set(TEST_SOURCES
      tests/test_main.cpp tests/test_bitboard.cpp tests/test_position.cpp
      tests/test_movegen.cpp tests/test_search.cpp tests/test_metal.cpp)

  add_executable(
    metalfish_tests
    ${TEST_SOURCES}
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES}
    ${SYZYGY_SOURCES}
    ${METAL_SOURCES})
  target_link_libraries(metalfish_tests Threads::Threads)

  if(APPLE)
    target_link_libraries(
      metalfish_tests ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
  endif()

  add_test(NAME metalfish_tests COMMAND metalfish_tests)
endif()
