/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  CUDA Graphs Implementation
*/

#ifdef USE_CUDA

#include "cuda_graphs.h"
#include <iostream>

namespace MetalFish {
namespace GPU {
namespace CUDA {

GraphManager::~GraphManager() {
  clear_all();
}

bool GraphManager::begin_capture(cudaStream_t stream, const std::string& name) {
  if (has_graph(name)) {
    std::cerr << "[CUDA Graphs] Graph '" << name << "' already exists" << std::endl;
    return false;
  }

  cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    std::cerr << "[CUDA Graphs] Failed to begin capture: " 
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  current_capture_name_ = name;
  return true;
}

bool GraphManager::end_capture(cudaStream_t stream, const std::string& name) {
  if (current_capture_name_ != name) {
    std::cerr << "[CUDA Graphs] Capture name mismatch" << std::endl;
    cudaStreamEndCapture(stream, nullptr);  // Abort capture
    return false;
  }

  GraphData data;
  cudaError_t err = cudaStreamEndCapture(stream, &data.graph);
  if (err != cudaSuccess) {
    std::cerr << "[CUDA Graphs] Failed to end capture: " 
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  // Get node count
  cudaGraphGetNodes(data.graph, nullptr, &data.node_count);

  // Instantiate the graph for execution
  err = cudaGraphInstantiate(&data.exec, data.graph, nullptr, nullptr, 0);
  if (err != cudaSuccess) {
    std::cerr << "[CUDA Graphs] Failed to instantiate graph: "
              << cudaGetErrorString(err) << std::endl;
    cudaGraphDestroy(data.graph);
    return false;
  }

  graphs_[name] = data;
  current_capture_name_.clear();

  std::cout << "[CUDA Graphs] Captured '" << name << "' with " 
            << data.node_count << " nodes" << std::endl;
  return true;
}

bool GraphManager::launch_graph(const std::string& name, cudaStream_t stream) {
  auto it = graphs_.find(name);
  if (it == graphs_.end()) {
    std::cerr << "[CUDA Graphs] Graph '" << name << "' not found" << std::endl;
    return false;
  }

  cudaError_t err = cudaGraphLaunch(it->second.exec, stream);
  if (err != cudaSuccess) {
    std::cerr << "[CUDA Graphs] Failed to launch graph: "
              << cudaGetErrorString(err) << std::endl;
    return false;
  }

  return true;
}

bool GraphManager::has_graph(const std::string& name) const {
  return graphs_.find(name) != graphs_.end();
}

void GraphManager::remove_graph(const std::string& name) {
  auto it = graphs_.find(name);
  if (it != graphs_.end()) {
    if (it->second.exec) {
      cudaGraphExecDestroy(it->second.exec);
    }
    if (it->second.graph) {
      cudaGraphDestroy(it->second.graph);
    }
    graphs_.erase(it);
  }
}

void GraphManager::clear_all() {
  for (auto& [name, data] : graphs_) {
    if (data.exec) {
      cudaGraphExecDestroy(data.exec);
    }
    if (data.graph) {
      cudaGraphDestroy(data.graph);
    }
  }
  graphs_.clear();
}

GraphManager::GraphStats GraphManager::get_stats() const {
  GraphStats stats{0, 0};
  stats.num_graphs = graphs_.size();
  for (const auto& [name, data] : graphs_) {
    stats.total_nodes += data.node_count;
  }
  return stats;
}

} // namespace CUDA
} // namespace GPU
} // namespace MetalFish

#endif // USE_CUDA
