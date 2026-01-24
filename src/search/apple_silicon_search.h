/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Apple Silicon Search Optimizations
  
  This header provides optimizations specifically tuned for Apple Silicon chips.
  All parameters are dynamically determined based on hardware detection.
  
  Key optimizations:
  - Cache-line aligned structures (128 bytes for M-series)
  - Optimal thread counts based on P/E core topology
  - Memory prefetching tuned for unified memory
  - SIMD-friendly data layouts (32-wide for Apple GPUs)
  - Dynamic batch sizing based on GPU core count
  
  Licensed under GPL-3.0
*/

#ifndef APPLE_SILICON_SEARCH_H_INCLUDED
#define APPLE_SILICON_SEARCH_H_INCLUDED

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <pthread.h>
#include <TargetConditionals.h>
#endif

namespace MetalFish {
namespace AppleSilicon {

// ============================================================================
// Hardware Detection - Dynamically determined at runtime
// ============================================================================

struct HardwareInfo {
  // CPU Core topology
  int performance_cores = 0;      // P-cores (high performance)
  int efficiency_cores = 0;       // E-cores (power efficient)
  int total_cores = 0;
  
  // Cache hierarchy
  size_t l1_cache_size = 0;       // L1 data cache per core
  size_t l2_cache_size = 0;       // L2 cache (shared on clusters)
  size_t cache_line_size = 128;   // 128 bytes for Apple Silicon
  
  // Memory system
  size_t total_memory = 0;
  size_t page_size = 16384;       // 16KB pages on Apple Silicon
  bool has_unified_memory = true;
  
  // Performance characteristics
  int memory_bandwidth_gbps = 0;  // Memory bandwidth in GB/s
  int chip_generation = 0;        // 1=M1, 2=M2, 3=M3, 4=M4
  bool is_pro_max_ultra = false;  // Higher-end variant
  
  // Computed optimal values
  int optimal_search_threads = 0;
  size_t optimal_tt_size_mb = 0;
  size_t optimal_hash_entries = 0;
  int prefetch_distance = 0;
};

// ============================================================================
// Runtime Detection Functions
// ============================================================================

#ifdef __APPLE__

namespace detail {

inline int get_sysctl_int(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  sysctlbyname(name, &value, &size, nullptr, 0);
  return value;
}

inline uint64_t get_sysctl_uint64(const char* name) {
  uint64_t value = 0;
  size_t size = sizeof(value);
  sysctlbyname(name, &value, &size, nullptr, 0);
  return value;
}

inline std::string get_sysctl_string(const char* name) {
  char buffer[256] = {0};
  size_t size = sizeof(buffer);
  if (sysctlbyname(name, buffer, &size, nullptr, 0) == 0) {
    return std::string(buffer, size > 0 ? size - 1 : 0);
  }
  return "";
}

// Detect chip generation from brand string
inline int detect_chip_generation(const std::string& brand) {
  if (brand.find("M4") != std::string::npos) return 4;
  if (brand.find("M3") != std::string::npos) return 3;
  if (brand.find("M2") != std::string::npos) return 2;
  if (brand.find("M1") != std::string::npos) return 1;
  return 0;
}

inline bool is_high_end_variant(const std::string& brand) {
  return brand.find("Pro") != std::string::npos ||
         brand.find("Max") != std::string::npos ||
         brand.find("Ultra") != std::string::npos;
}

// Estimate memory bandwidth based on chip variant
inline int estimate_bandwidth(int generation, bool high_end, size_t total_mem) {
  // Base bandwidth values (GB/s)
  int base = 100;
  
  if (total_mem >= 192ULL * 1024 * 1024 * 1024) {
    // Ultra variant
    base = generation >= 3 ? 800 : 400;
  } else if (total_mem >= 64ULL * 1024 * 1024 * 1024) {
    // Max variant
    base = generation >= 3 ? 400 : 300;
  } else if (total_mem >= 32ULL * 1024 * 1024 * 1024) {
    // Pro variant
    base = generation >= 3 ? 200 : 150;
  } else {
    // Base variant
    base = generation >= 3 ? 100 : 68;
  }
  
  return base;
}

} // namespace detail

inline HardwareInfo detect_hardware() {
  HardwareInfo info;
  
  // Get CPU brand string
  std::string brand = detail::get_sysctl_string("machdep.cpu.brand_string");
  
  // Detect chip generation
  info.chip_generation = detail::detect_chip_generation(brand);
  info.is_pro_max_ultra = detail::is_high_end_variant(brand);
  
  // Get core counts
  // perflevel0 = performance cores, perflevel1 = efficiency cores
  info.performance_cores = detail::get_sysctl_int("hw.perflevel0.physicalcpu");
  info.efficiency_cores = detail::get_sysctl_int("hw.perflevel1.physicalcpu");
  info.total_cores = detail::get_sysctl_int("hw.physicalcpu");
  
  // Fallback if perflevel not available
  if (info.performance_cores == 0 && info.efficiency_cores == 0) {
    info.total_cores = detail::get_sysctl_int("hw.ncpu");
    // Estimate: typically 50-60% P-cores on Apple Silicon
    info.performance_cores = (info.total_cores * 3 + 2) / 5;
    info.efficiency_cores = info.total_cores - info.performance_cores;
  }
  
  // Get cache information
  info.l1_cache_size = detail::get_sysctl_uint64("hw.l1dcachesize");
  info.l2_cache_size = detail::get_sysctl_uint64("hw.l2cachesize");
  info.cache_line_size = detail::get_sysctl_int("hw.cachelinesize");
  
  // Default cache line size for Apple Silicon is 128 bytes
  if (info.cache_line_size == 0) {
    info.cache_line_size = 128;
  }
  
  // Get memory information
  info.total_memory = detail::get_sysctl_uint64("hw.memsize");
  info.page_size = detail::get_sysctl_int("hw.pagesize");
  if (info.page_size == 0) info.page_size = 16384; // 16KB default
  
  // Unified memory is always true for Apple Silicon
  info.has_unified_memory = true;
  
  // Estimate memory bandwidth
  info.memory_bandwidth_gbps = detail::estimate_bandwidth(
    info.chip_generation, info.is_pro_max_ultra, info.total_memory);
  
  // Calculate optimal search threads
  // For chess search, P-cores are most effective
  // Use P-cores for main search, E-cores can help with parallel work
  // But too many threads cause contention
  if (info.total_memory >= 192ULL * 1024 * 1024 * 1024) {
    // Ultra: can use many threads effectively
    info.optimal_search_threads = std::min(info.performance_cores + info.efficiency_cores / 2, 24);
  } else if (info.total_memory >= 64ULL * 1024 * 1024 * 1024) {
    // Max: good parallelism
    info.optimal_search_threads = std::min(info.performance_cores + info.efficiency_cores / 4, 16);
  } else if (info.is_pro_max_ultra) {
    // Pro: balanced
    info.optimal_search_threads = std::min(info.performance_cores + 2, 12);
  } else {
    // Base: focus on P-cores
    info.optimal_search_threads = std::min(info.performance_cores + 1, 8);
  }
  
  // Calculate optimal TT size
  // Reserve memory for: OS, NNUE networks (~100MB), evaluation caches, etc.
  size_t reserved_mb = 512 + (info.total_memory / (8ULL * 1024 * 1024 * 1024)) * 256;
  size_t available_mb = (info.total_memory / (1024 * 1024)) - reserved_mb;
  
  // Use 50-75% of available memory for TT depending on total memory
  float tt_fraction = info.total_memory >= 32ULL * 1024 * 1024 * 1024 ? 0.6f : 0.5f;
  info.optimal_tt_size_mb = static_cast<size_t>(available_mb * tt_fraction);
  
  // Round down to power of 2 for efficient hashing
  size_t power = 1;
  while (power * 2 <= info.optimal_tt_size_mb) {
    power *= 2;
  }
  info.optimal_tt_size_mb = power;
  
  // Cap at reasonable maximum (32GB for TT)
  info.optimal_tt_size_mb = std::min(info.optimal_tt_size_mb, size_t(32768));
  
  // Calculate optimal hash entries (each cluster is 32 bytes)
  info.optimal_hash_entries = (info.optimal_tt_size_mb * 1024 * 1024) / 32;
  
  // Prefetch distance based on memory bandwidth and latency
  // Higher bandwidth = can prefetch further ahead
  info.prefetch_distance = 2 + info.memory_bandwidth_gbps / 100;
  info.prefetch_distance = std::min(info.prefetch_distance, 8);
  
  return info;
}

// Cached hardware info (computed once at startup)
inline const HardwareInfo& get_hardware_info() {
  static HardwareInfo info = detect_hardware();
  return info;
}

#else // Non-Apple platforms

inline HardwareInfo detect_hardware() {
  HardwareInfo info;
  info.performance_cores = 4;
  info.efficiency_cores = 0;
  info.total_cores = 4;
  info.l1_cache_size = 32 * 1024;
  info.l2_cache_size = 256 * 1024;
  info.cache_line_size = 64;
  info.total_memory = 8ULL * 1024 * 1024 * 1024;
  info.page_size = 4096;
  info.has_unified_memory = false;
  info.memory_bandwidth_gbps = 50;
  info.chip_generation = 0;
  info.is_pro_max_ultra = false;
  info.optimal_search_threads = 4;
  info.optimal_tt_size_mb = 256;
  info.optimal_hash_entries = 8 * 1024 * 1024;
  info.prefetch_distance = 2;
  return info;
}

inline const HardwareInfo& get_hardware_info() {
  static HardwareInfo info = detect_hardware();
  return info;
}

#endif // __APPLE__

// ============================================================================
// Cache-Aligned Allocator
// ============================================================================

// Alignment for Apple Silicon cache lines (128 bytes)
constexpr size_t APPLE_CACHE_LINE = 128;

// Align to Apple Silicon cache line
template<typename T>
struct alignas(APPLE_CACHE_LINE) CacheAligned {
  T value;
  
  CacheAligned() = default;
  CacheAligned(const T& v) : value(v) {}
  CacheAligned& operator=(const T& v) { value = v; return *this; }
  operator T&() { return value; }
  operator const T&() const { return value; }
};

// ============================================================================
// Memory Prefetching Utilities
// ============================================================================

// Prefetch for read (temporal - keep in cache)
inline void prefetch_read(const void* addr) {
#ifdef __APPLE__
  __builtin_prefetch(addr, 0, 3);
#endif
}

// Prefetch for write (temporal - keep in cache)
inline void prefetch_write(void* addr) {
#ifdef __APPLE__
  __builtin_prefetch(addr, 1, 3);
#endif
}

// Prefetch for read (non-temporal - don't pollute cache)
inline void prefetch_read_nt(const void* addr) {
#ifdef __APPLE__
  __builtin_prefetch(addr, 0, 0);
#endif
}

// Prefetch multiple cache lines ahead
template<int N = 1>
inline void prefetch_ahead(const void* base, size_t offset) {
  const size_t cache_line = get_hardware_info().cache_line_size;
  for (int i = 0; i < N; ++i) {
    prefetch_read(static_cast<const char*>(base) + offset + i * cache_line);
  }
}

// ============================================================================
// Thread Affinity Helpers
// ============================================================================

#ifdef __APPLE__

// Set thread to prefer performance cores
inline bool set_thread_performance_priority() {
  pthread_t thread = pthread_self();
  
  // Use QoS class to hint scheduler
  // QOS_CLASS_USER_INTERACTIVE runs on P-cores
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  
  return true;
}

// Set thread to prefer efficiency cores (for background work)
inline bool set_thread_efficiency_priority() {
  pthread_t thread = pthread_self();
  
  // QOS_CLASS_UTILITY runs on E-cores when possible
  pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
  
  return true;
}

// Set thread to balanced priority
inline bool set_thread_balanced_priority() {
  pthread_t thread = pthread_self();
  
  // QOS_CLASS_USER_INITIATED is balanced
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
  
  return true;
}

#else

inline bool set_thread_performance_priority() { return false; }
inline bool set_thread_efficiency_priority() { return false; }
inline bool set_thread_balanced_priority() { return false; }

#endif

// ============================================================================
// Atomic Operations Optimized for Apple Silicon
// ============================================================================

// Apple Silicon has strong memory ordering, so we can use relaxed atomics
// more aggressively for better performance

template<typename T>
inline T atomic_load_relaxed(const std::atomic<T>& a) {
  return a.load(std::memory_order_relaxed);
}

template<typename T>
inline void atomic_store_relaxed(std::atomic<T>& a, T value) {
  a.store(value, std::memory_order_relaxed);
}

// For statistics that don't need strict ordering
template<typename T>
inline void atomic_add_relaxed(std::atomic<T>& a, T value) {
  a.fetch_add(value, std::memory_order_relaxed);
}

// ============================================================================
// SIMD-Friendly Data Layout Helpers
// ============================================================================

// Apple GPUs use 32-wide SIMD groups
constexpr int SIMD_WIDTH = 32;

// Round up to SIMD width for efficient GPU processing
constexpr size_t align_to_simd(size_t n) {
  return (n + SIMD_WIDTH - 1) & ~(SIMD_WIDTH - 1);
}

// ============================================================================
// Memory Pressure Monitoring
// ============================================================================

#ifdef __APPLE__

inline float get_memory_pressure() {
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  vm_statistics64_data_t vm_stat;
  
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64,
                        reinterpret_cast<host_info64_t>(&vm_stat),
                        &count) != KERN_SUCCESS) {
    return 0.0f;
  }
  
  uint64_t total = vm_stat.free_count + vm_stat.active_count + 
                   vm_stat.inactive_count + vm_stat.wire_count;
  if (total == 0) return 0.0f;
  
  float pressure = static_cast<float>(vm_stat.active_count + vm_stat.wire_count) /
                   static_cast<float>(total);
  
  return std::min(1.0f, std::max(0.0f, pressure));
}

inline bool should_reduce_memory_usage() {
  return get_memory_pressure() > 0.85f;
}

#else

inline float get_memory_pressure() { return 0.0f; }
inline bool should_reduce_memory_usage() { return false; }

#endif

// ============================================================================
// Search Parameter Tuning Based on Hardware
// ============================================================================

struct SearchTuning {
  // LMR parameters adjusted for hardware
  int lmr_base = 0;
  int lmr_divisor = 0;
  
  // Null move pruning
  int nmp_base_reduction = 0;
  int nmp_depth_divisor = 0;
  
  // Futility pruning margins
  int futility_margin_base = 0;
  int futility_margin_per_depth = 0;
  
  // Aspiration window
  int aspiration_delta = 0;
  
  // Time management
  float time_optimal_fraction = 0.0f;
  float time_maximum_fraction = 0.0f;
};

inline SearchTuning compute_search_tuning() {
  const auto& hw = get_hardware_info();
  SearchTuning tuning;
  
  // Adjust LMR based on core count
  // More cores = can search deeper, so slightly less aggressive reduction
  tuning.lmr_base = 77 + hw.performance_cores * 2;
  tuning.lmr_divisor = 235 + hw.performance_cores * 5;
  
  // NMP: more aggressive with more memory bandwidth
  tuning.nmp_base_reduction = 3 + hw.memory_bandwidth_gbps / 100;
  tuning.nmp_depth_divisor = 3;
  
  // Futility: adjust based on memory/compute balance
  tuning.futility_margin_base = 200 - hw.memory_bandwidth_gbps / 4;
  tuning.futility_margin_per_depth = 100;
  
  // Aspiration: tighter windows with more compute power
  tuning.aspiration_delta = std::max(10, 20 - hw.performance_cores);
  
  // Time management: can think longer with more cores
  tuning.time_optimal_fraction = 0.05f + 0.005f * hw.optimal_search_threads;
  tuning.time_maximum_fraction = 0.25f + 0.02f * hw.optimal_search_threads;
  
  return tuning;
}

inline const SearchTuning& get_search_tuning() {
  static SearchTuning tuning = compute_search_tuning();
  return tuning;
}

// ============================================================================
// Transposition Table Optimization Parameters
// ============================================================================

struct TTOptimization {
  // Cluster size (entries per bucket)
  int cluster_size = 3;
  
  // Replacement strategy parameters
  int age_weight = 8;
  int depth_weight = 1;
  
  // Prefetch settings
  int prefetch_clusters = 2;
  
  // Memory layout
  size_t cluster_alignment = 128;  // Apple Silicon cache line
};

inline TTOptimization compute_tt_optimization() {
  const auto& hw = get_hardware_info();
  TTOptimization opt;
  
  // Standard cluster size of 3 fits well in 32 bytes (with padding)
  opt.cluster_size = 3;
  
  // Age weight: higher with more memory (entries stay useful longer)
  opt.age_weight = 6 + static_cast<int>(hw.total_memory / (16ULL * 1024 * 1024 * 1024));
  opt.age_weight = std::min(opt.age_weight, 12);
  
  opt.depth_weight = 1;
  
  // Prefetch: more with higher bandwidth
  opt.prefetch_clusters = 1 + hw.memory_bandwidth_gbps / 150;
  opt.prefetch_clusters = std::min(opt.prefetch_clusters, 4);
  
  // Alignment to cache line
  opt.cluster_alignment = hw.cache_line_size;
  
  return opt;
}

inline const TTOptimization& get_tt_optimization() {
  static TTOptimization opt = compute_tt_optimization();
  return opt;
}

} // namespace AppleSilicon
} // namespace MetalFish

#endif // APPLE_SILICON_SEARCH_H_INCLUDED
