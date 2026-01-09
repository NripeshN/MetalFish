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

# Metal GPU acceleration option (macOS only, defaults to ON for Apple devices)
if(APPLE)
  option(USE_METAL "Enable Metal GPU acceleration" ON)
else()
  option(USE_METAL "Enable Metal GPU acceleration" OFF)
endif()

# Metal-cpp headers location
set(METAL_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/metal-cpp")
set(METAL_CPP_HEADER "${METAL_CPP_DIR}/Metal/Metal.hpp")

# Download metal-cpp if USE_METAL is ON and headers don't exist
if(APPLE AND USE_METAL)
  if(NOT EXISTS "${METAL_CPP_HEADER}")
    message(STATUS "metal-cpp headers not found, downloading...")
    
    # Create external directory if it doesn't exist
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/external")
    
    # Download metal-cpp from Apple's GitHub
    set(METAL_CPP_URL "https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip")
    set(METAL_CPP_ZIP "${CMAKE_CURRENT_BINARY_DIR}/metal-cpp.zip")
    set(METAL_CPP_EXTRACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/metal-cpp-extract")
    
    # Download
    file(DOWNLOAD 
      ${METAL_CPP_URL}
      ${METAL_CPP_ZIP}
      STATUS DOWNLOAD_STATUS
      SHOW_PROGRESS
    )
    list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
    
    if(NOT DOWNLOAD_RESULT EQUAL 0)
      message(WARNING "Failed to download metal-cpp. Metal support will be disabled.")
      set(USE_METAL OFF)
    else()
      # Extract
      message(STATUS "Extracting metal-cpp...")
      file(ARCHIVE_EXTRACT INPUT ${METAL_CPP_ZIP} DESTINATION ${METAL_CPP_EXTRACT_DIR})
      
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
        message(WARNING "Failed to extract metal-cpp. Metal support will be disabled.")
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
    message(WARNING "metal-cpp headers still not available. Metal support will be disabled.")
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

# Metal GPU acceleration source files (macOS only, when enabled)
set(METAL_SOURCES)
if(USE_METAL AND METAL_CPP_AVAILABLE)
  set(METAL_SOURCES src/metal/metal_device.mm src/metal/device.cpp
                    src/metal/allocator.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_METAL")
  message(STATUS "Metal GPU acceleration: ENABLED")
else()
  message(STATUS "Metal GPU acceleration: DISABLED")
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

  if(USE_METAL AND METAL_CPP_AVAILABLE)
    target_link_libraries(metalfish ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
                          ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
  endif()
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

  if(APPLE AND USE_METAL AND METAL_CPP_AVAILABLE)
    target_link_libraries(
      metalfish_tests ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
      ${COREFOUNDATION_FRAMEWORK} ${QUARTZCORE_FRAMEWORK})
  endif()

  add_test(NAME metalfish_tests COMMAND metalfish_tests)
endif()
