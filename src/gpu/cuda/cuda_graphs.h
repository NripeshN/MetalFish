/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Graphs Support

  Implements CUDA graphs for reduced kernel launch overhead.
  CUDA graphs capture a sequence of operations and replay them efficiently.
*/

#ifndef CUDA_GRAPHS_H
#define CUDA_GRAPHS_H

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace MetalFish {
namespace GPU {
namespace CUDA {

/**
 * CUDA Graph Manager
 * 
 * Captures and replays sequences of CUDA operations for improved performance.
 * Particularly useful for repetitive evaluation patterns in NNUE.
 */
class GraphManager {
public:
  GraphManager() = default;
  ~GraphManager();

  /**
   * Begin graph capture on a stream
   */
  bool begin_capture(cudaStream_t stream, const std::string& name);

  /**
   * End graph capture and store the graph
   */
  bool end_capture(cudaStream_t stream, const std::string& name);

  /**
   * Launch a captured graph
   */
  bool launch_graph(const std::string& name, cudaStream_t stream);

  /**
   * Check if a graph exists
   */
  bool has_graph(const std::string& name) const;

  /**
   * Delete a graph
   */
  void remove_graph(const std::string& name);

  /**
   * Clear all graphs
   */
  void clear_all();

  /**
   * Get graph statistics
   */
  struct GraphStats {
    size_t num_graphs;
    size_t total_nodes;
  };
  GraphStats get_stats() const;

private:
  struct GraphData {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    size_t node_count = 0;
  };

  std::unordered_map<std::string, GraphData> graphs_;
  std::string current_capture_name_;
};

/**
 * RAII helper for graph capture
 */
class ScopedGraphCapture {
public:
  ScopedGraphCapture(GraphManager& manager, cudaStream_t stream, 
                     const std::string& name)
      : manager_(manager), stream_(stream), name_(name), active_(false) {
    active_ = manager_.begin_capture(stream_, name_);
  }

  ~ScopedGraphCapture() {
    if (active_) {
      manager_.end_capture(stream_, name_);
    }
  }

  bool is_active() const { return active_; }

private:
  GraphManager& manager_;
  cudaStream_t stream_;
  std::string name_;
  bool active_;
};

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
#endif // CUDA_GRAPHS_H
