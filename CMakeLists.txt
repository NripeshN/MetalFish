# MetalFish - GPU-accelerated Chess Engine Copyright (c) 2025 Nripesh Niketan
# Licensed under GPL-3.0

cmake_minimum_required(VERSION 3.20)
project(
  metalfish
  VERSION 1.0.0
  LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# macOS specific settings
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET
      "14.0"
      CACHE STRING "Minimum macOS deployment version")

  # Find Metal framework
  find_library(METAL_LIBRARY Metal REQUIRED)
  find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
  find_library(QUARTZCORE_LIBRARY QuartzCore REQUIRED)

  message(STATUS "Metal library: ${METAL_LIBRARY}")
  message(STATUS "Foundation library: ${FOUNDATION_LIBRARY}")
  message(STATUS "QuartzCore library: ${QUARTZCORE_LIBRARY}")
endif()

# Metal-cpp header-only library path
set(METAL_CPP_DIR "${CMAKE_SOURCE_DIR}/external/metal-cpp")

# Download metal-cpp if not present
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(STATUS "Downloading metal-cpp...")
  file(DOWNLOAD
       "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS15_iOS18.zip"
       "${CMAKE_BINARY_DIR}/metal-cpp.zip" SHOW_PROGRESS)
  file(ARCHIVE_EXTRACT INPUT "${CMAKE_BINARY_DIR}/metal-cpp.zip" DESTINATION
       "${CMAKE_BINARY_DIR}/metal-cpp-extract")
  # Find the extracted directory
  file(GLOB METAL_CPP_EXTRACTED "${CMAKE_BINARY_DIR}/metal-cpp-extract/*")
  if(METAL_CPP_EXTRACTED AND NOT EXISTS "${METAL_CPP_DIR}")
    file(RENAME "${METAL_CPP_EXTRACTED}" "${METAL_CPP_DIR}")
  endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include ${METAL_CPP_DIR})

# Source files - Metal backend
set(METAL_SOURCES src/metal/device.cpp src/metal/allocator.cpp
                  src/metal/kernels.cpp src/metal/gpu_ops.cpp)

# Source files - Core chess engine
set(CORE_SOURCES src/core/bitboard.cpp src/core/position.cpp
                 src/core/movegen.cpp src/core/zobrist.cpp src/core/perft.cpp)

# Source files - Search
set(SEARCH_SOURCES src/search/search.cpp src/search/thread.cpp
                   src/search/tt.cpp src/search/movepick.cpp
                   src/search/batch_eval.cpp)

# Source files - Evaluation
set(EVAL_SOURCES src/eval/evaluate.cpp src/eval/nnue.cpp src/eval/nnue_gpu.cpp
                 src/eval/gpu_nnue.cpp src/eval/nnue_loader.cpp)

# Source files - UCI
set(UCI_SOURCES src/uci/uci.cpp src/uci/benchmark.cpp)

# Metal shaders
set(METAL_SHADERS shaders/nnue_stockfish.metal shaders/nnue.metal
                  shaders/core_ops.metal)

# Option to build Metal shaders (requires Metal toolchain)
option(BUILD_METAL_SHADERS "Build Metal shaders (requires xcrun metal)" OFF)
option(BUILD_TESTS "Build test suite" ON)

# Compile Metal shaders to .metallib
if(APPLE AND BUILD_METAL_SHADERS)
  set(METALLIB_OUTPUT "${CMAKE_BINARY_DIR}/metalfish.metallib")

  # First compile to .air files
  set(AIR_FILES "")
  foreach(SHADER ${METAL_SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    set(AIR_FILE "${CMAKE_BINARY_DIR}/${SHADER_NAME}.air")
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal -c "${CMAKE_SOURCE_DIR}/${SHADER}" -o
              ${AIR_FILE}
      DEPENDS "${CMAKE_SOURCE_DIR}/${SHADER}"
      COMMENT "Compiling ${SHADER} to AIR")
    list(APPEND AIR_FILES ${AIR_FILE})
  endforeach()

  # Link .air files into .metallib
  add_custom_command(
    OUTPUT ${METALLIB_OUTPUT}
    COMMAND xcrun -sdk macosx metallib ${AIR_FILES} -o ${METALLIB_OUTPUT}
    DEPENDS ${AIR_FILES}
    COMMENT "Linking Metal shaders into metallib")

  add_custom_target(metal_shaders DEPENDS ${METALLIB_OUTPUT})
elseif(APPLE)
  message(
    STATUS
      "Metal shader compilation disabled. Enable with -DBUILD_METAL_SHADERS=ON")
  # Create placeholder metallib path
  set(METALLIB_OUTPUT "${CMAKE_BINARY_DIR}/metalfish.metallib")
endif()

# Main executable
add_executable(metalfish src/main.cpp ${METAL_SOURCES} ${CORE_SOURCES}
                         ${SEARCH_SOURCES} ${EVAL_SOURCES} ${UCI_SOURCES})

if(APPLE)
  if(BUILD_METAL_SHADERS AND TARGET metal_shaders)
    add_dependencies(metalfish metal_shaders)

    # Copy metallib to output directory
    add_custom_command(
      TARGET metalfish
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${METALLIB_OUTPUT}
              $<TARGET_FILE_DIR:metalfish>/
      COMMENT "Copying metallib to output directory")
  endif()

  target_link_libraries(
    metalfish ${METAL_LIBRARY} ${FOUNDATION_LIBRARY} ${QUARTZCORE_LIBRARY}
    "-framework MetalPerformanceShaders")

  # Define Metal path for runtime
  target_compile_definitions(
    metalfish PRIVATE METAL_PATH="${CMAKE_BINARY_DIR}/metalfish.metallib")
endif()

# Compiler flags
target_compile_options(metalfish PRIVATE -Wall -Wextra -O3 -DNDEBUG
                                         $<$<CONFIG:Debug>:-g -O0>)

# Test executable
if(BUILD_TESTS)
  add_executable(
    metalfish_tests
    tests/test_main.cpp
    tests/test_bitboard.cpp
    tests/test_position.cpp
    tests/test_movegen.cpp
    tests/test_search.cpp
    tests/test_metal.cpp
    ${METAL_SOURCES}
    ${CORE_SOURCES}
    ${SEARCH_SOURCES}
    ${EVAL_SOURCES}
    ${UCI_SOURCES})

  target_include_directories(metalfish_tests PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                     ${METAL_CPP_DIR})

  if(APPLE)
    target_link_libraries(
      metalfish_tests ${METAL_LIBRARY} ${FOUNDATION_LIBRARY}
      ${QUARTZCORE_LIBRARY} "-framework MetalPerformanceShaders")

    target_compile_definitions(
      metalfish_tests
      PRIVATE METAL_PATH="${CMAKE_BINARY_DIR}/metalfish.metallib")
  endif()

  target_compile_options(metalfish_tests PRIVATE -Wall -Wextra -g)

  # Enable CTest
  enable_testing()
  add_test(NAME MetalFishTests COMMAND metalfish_tests)
endif()

# Install
install(TARGETS metalfish DESTINATION bin)
if(EXISTS ${METALLIB_OUTPUT})
  install(FILES ${METALLIB_OUTPUT} DESTINATION lib)
endif()

# Print configuration summary
message(STATUS "")
message(STATUS "MetalFish Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  macOS Deployment Target: ${CMAKE_OSX_DEPLOYMENT_TARGET}")
message(STATUS "  Build Tests: ${BUILD_TESTS}")
message(STATUS "  Build Metal Shaders: ${BUILD_METAL_SHADERS}")
message(STATUS "")
