/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Profiling Infrastructure

  Profiling utilities including:
  - NVTX markers for Nsight profiling
  - Kernel timing
  - Occupancy calculator
  - Performance metrics collection
*/

#ifndef CUDA_PROFILING_H
#define CUDA_PROFILING_H

#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// NVTX profiling support (optional)
#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace MetalFish {
namespace GPU {
namespace CUDA {

// ============================================================================
// NVTX Markers (for Nsight profiling)
// ============================================================================

class NVTXMarker {
public:
#ifdef USE_NVTX
  NVTXMarker(const char *name, uint32_t color = 0xFF00FF00) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    
    nvtxRangePushEx(&eventAttrib);
  }
  
  ~NVTXMarker() {
    nvtxRangePop();
  }
#else
  NVTXMarker(const char *, uint32_t = 0) {}
  ~NVTXMarker() {}
#endif
};

// Convenience macro
#define NVTX_RANGE(name) NVTXMarker _nvtx_marker(name)
#define NVTX_RANGE_COLOR(name, color) NVTXMarker _nvtx_marker(name, color)

// ============================================================================
// Kernel Timer
// ============================================================================

class KernelTimer {
public:
  KernelTimer(const std::string &name, cudaStream_t stream = 0)
      : name_(name), stream_(stream) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    cudaEventRecord(start_event_, stream_);
  }
  
  ~KernelTimer() {
    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start_event_, stop_event_);
    
    // Record timing
    timings_[name_].push_back(ms);
    
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
  }
  
  // Get average time for a kernel
  static float get_average_time(const std::string &name) {
    auto it = timings_.find(name);
    if (it == timings_.end() || it->second.empty()) {
      return 0.0f;
    }
    
    float sum = 0.0f;
    for (float t : it->second) {
      sum += t;
    }
    return sum / it->second.size();
  }
  
  // Print all timing statistics
  static void print_stats() {
    std::cout << "\n[CUDA Kernel Timing Statistics]" << std::endl;
    std::cout << "======================================" << std::endl;
    
    for (const auto &[name, times] : timings_) {
      if (times.empty()) continue;
      
      float sum = 0.0f, min_time = times[0], max_time = times[0];
      for (float t : times) {
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
      }
      float avg = sum / times.size();
      
      std::cout << name << ":" << std::endl;
      std::cout << "  Calls:   " << times.size() << std::endl;
      std::cout << "  Average: " << avg << " ms" << std::endl;
      std::cout << "  Min:     " << min_time << " ms" << std::endl;
      std::cout << "  Max:     " << max_time << " ms" << std::endl;
      std::cout << "  Total:   " << sum << " ms" << std::endl;
    }
  }
  
  // Reset all timings
  static void reset() {
    timings_.clear();
  }
  
private:
  std::string name_;
  cudaStream_t stream_;
  cudaEvent_t start_event_;
  cudaEvent_t stop_event_;
  
  static std::map<std::string, std::vector<float>> timings_;
};

// Convenience macro
#define TIME_KERNEL(name, stream) KernelTimer _kernel_timer(name, stream)

// ============================================================================
// Occupancy Calculator
// ============================================================================

class OccupancyCalculator {
public:
  /**
   * Calculate theoretical occupancy for a kernel
   */
  static float calculate_occupancy(const void *kernel, int block_size, 
                                   size_t dynamic_smem_size = 0) {
    int min_grid_size, optimal_block_size;
    
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                       kernel, dynamic_smem_size, 0);
    
    // Get device properties
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    // Calculate occupancy
    int max_active_blocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, kernel,
                                                   block_size, dynamic_smem_size);
    
    float occupancy = (max_active_blocks * block_size / 
                      static_cast<float>(prop.maxThreadsPerMultiProcessor));
    
    return occupancy;
  }
  
  /**
   * Print occupancy information for a kernel
   */
  static void print_occupancy_info(const std::string &name, const void *kernel,
                                   int block_size, size_t dynamic_smem_size = 0) {
    float occupancy = calculate_occupancy(kernel, block_size, dynamic_smem_size);
    
    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel);
    
    std::cout << "\n[Occupancy Info: " << name << "]" << std::endl;
    std::cout << "  Block Size:        " << block_size << std::endl;
    std::cout << "  Registers/Thread:  " << attr.numRegs << std::endl;
    std::cout << "  Shared Mem:        " << (attr.sharedSizeBytes + dynamic_smem_size) << " bytes" << std::endl;
    std::cout << "  Occupancy:         " << (occupancy * 100.0f) << "%" << std::endl;
    
    // Suggest optimal block size
    int min_grid_size, optimal_block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                       kernel, dynamic_smem_size, 0);
    std::cout << "  Optimal Block Size: " << optimal_block_size << std::endl;
  }
  
  /**
   * Auto-tune block size for best occupancy
   */
  static int find_optimal_block_size(const void *kernel, 
                                     size_t dynamic_smem_size = 0) {
    int min_grid_size, optimal_block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &optimal_block_size,
                                       kernel, dynamic_smem_size, 0);
    return optimal_block_size;
  }
};

// ============================================================================
// Performance Metrics Collector
// ============================================================================

class PerformanceMetrics {
public:
  struct Metrics {
    float kernel_time_ms = 0.0f;
    float memory_throughput_gbps = 0.0f;
    float compute_throughput_gflops = 0.0f;
    float occupancy = 0.0f;
    size_t memory_transferred = 0;
  };
  
  /**
   * Measure kernel performance
   */
  static Metrics measure_kernel(const std::string &name, 
                                std::function<void()> kernel_launch,
                                size_t memory_transferred = 0,
                                size_t flops = 0) {
    Metrics m;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm-up
    kernel_launch();
    cudaDeviceSynchronize();
    
    // Measure
    cudaEventRecord(start);
    kernel_launch();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&m.kernel_time_ms, start, stop);
    
    // Calculate throughput
    if (memory_transferred > 0 && m.kernel_time_ms > 0) {
      float seconds = m.kernel_time_ms / 1000.0f;
      m.memory_throughput_gbps = (memory_transferred / 1e9) / seconds;
    }
    
    if (flops > 0 && m.kernel_time_ms > 0) {
      float seconds = m.kernel_time_ms / 1000.0f;
      m.compute_throughput_gflops = (flops / 1e9) / seconds;
    }
    
    m.memory_transferred = memory_transferred;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Store metrics
    metrics_[name] = m;
    
    return m;
  }
  
  /**
   * Print performance report
   */
  static void print_report() {
    std::cout << "\n[CUDA Performance Report]" << std::endl;
    std::cout << "================================================" << std::endl;
    
    for (const auto &[name, m] : metrics_) {
      std::cout << name << ":" << std::endl;
      std::cout << "  Time:               " << m.kernel_time_ms << " ms" << std::endl;
      if (m.memory_throughput_gbps > 0) {
        std::cout << "  Memory Throughput:  " << m.memory_throughput_gbps << " GB/s" << std::endl;
      }
      if (m.compute_throughput_gflops > 0) {
        std::cout << "  Compute Throughput: " << m.compute_throughput_gflops << " GFLOPS" << std::endl;
      }
      if (m.occupancy > 0) {
        std::cout << "  Occupancy:          " << (m.occupancy * 100.0f) << "%" << std::endl;
      }
      std::cout << std::endl;
    }
  }
  
  static void reset() {
    metrics_.clear();
  }
  
private:
  static std::map<std::string, Metrics> metrics_;
};

// ============================================================================
// CPU Timer (for comparison)
// ============================================================================

class CPUTimer {
public:
  CPUTimer(const std::string &name) 
      : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
  
  ~CPUTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
    
    std::cout << "[CPU Timer] " << name_ << ": " 
              << (duration.count() / 1000.0) << " ms" << std::endl;
  }
  
private:
  std::string name_;
  std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Bandwidth Tester
// ============================================================================

class BandwidthTester {
public:
  /**
   * Measure host to device bandwidth
   */
  static float measure_h2d_bandwidth(size_t size) {
    void *h_data, *d_data;
    cudaMallocHost(&h_data, size);
    cudaMalloc(&d_data, size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float bandwidth_gbps = (size / 1e9) / (ms / 1000.0f);
    
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return bandwidth_gbps;
  }
  
  /**
   * Measure device to host bandwidth
   */
  static float measure_d2h_bandwidth(size_t size) {
    void *h_data, *d_data;
    cudaMallocHost(&h_data, size);
    cudaMalloc(&d_data, size);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    float bandwidth_gbps = (size / 1e9) / (ms / 1000.0f);
    
    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return bandwidth_gbps;
  }
  
  /**
   * Print bandwidth test results
   */
  static void print_bandwidth_tests() {
    std::cout << "\n[CUDA Bandwidth Tests]" << std::endl;
    std::cout << "================================" << std::endl;
    
    std::vector<size_t> sizes = {
      1 * 1024 * 1024,      // 1 MB
      16 * 1024 * 1024,     // 16 MB
      64 * 1024 * 1024,     // 64 MB
      256 * 1024 * 1024     // 256 MB
    };
    
    for (size_t size : sizes) {
      float h2d = measure_h2d_bandwidth(size);
      float d2h = measure_d2h_bandwidth(size);
      
      std::cout << "Size: " << (size / (1024 * 1024)) << " MB" << std::endl;
      std::cout << "  H2D: " << h2d << " GB/s" << std::endl;
      std::cout << "  D2H: " << d2h << " GB/s" << std::endl;
    }
  }
};

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

// Initialize static members
namespace MetalFish {
namespace GPU {
namespace CUDA {
std::map<std::string, std::vector<float>> KernelTimer::timings_;
std::map<std::string, PerformanceMetrics::Metrics> PerformanceMetrics::metrics_;
} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // CUDA_PROFILING_H
