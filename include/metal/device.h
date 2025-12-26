/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  MetalFish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

namespace MetalFish {
namespace Metal {

// Forward declarations
struct CommandEncoder;
struct DeviceStream;

// Function constant list type for kernel specialization
using MTLFCList = std::vector<std::tuple<const void*, MTL::DataType, NS::UInteger>>;

/**
 * CommandEncoder wraps MTL::ComputeCommandEncoder with memory management
 * and barrier handling for efficient GPU command submission.
 */
struct CommandEncoder {
    explicit CommandEncoder(DeviceStream& stream);
    CommandEncoder(const CommandEncoder&) = delete;
    CommandEncoder& operator=(const CommandEncoder&) = delete;
    ~CommandEncoder();

    // Set buffer at index
    void set_buffer(const MTL::Buffer* buf, int idx, int64_t offset = 0);
    
    // Set bytes directly (for small uniform data)
    template <typename T>
    void set_bytes(const T& value, int idx) {
        enc_->setBytes(&value, sizeof(T), idx);
    }
    
    template <typename T>
    void set_bytes(const T* data, size_t count, int idx) {
        enc_->setBytes(data, count * sizeof(T), idx);
    }

    // Dispatch compute work
    void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
    void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
    
    // Set compute pipeline state
    void set_compute_pipeline_state(MTL::ComputePipelineState* kernel);
    
    // Set threadgroup memory
    void set_threadgroup_memory_length(size_t length, int idx);
    
    // Memory barrier
    void barrier();

private:
    DeviceStream& stream_;
    MTL::ComputeCommandEncoder* enc_;
    bool needs_barrier_{false};
};

/**
 * DeviceStream manages command buffers and encoders for a command queue.
 */
struct DeviceStream {
    explicit DeviceStream(MTL::CommandQueue* queue);
    ~DeviceStream();
    
    MTL::CommandQueue* queue;
    MTL::CommandBuffer* buffer{nullptr};
    std::unique_ptr<CommandEncoder> encoder{nullptr};
    int buffer_ops{0};
};

/**
 * Device is the main interface to the Metal GPU device.
 * It handles device initialization, kernel compilation, and command submission.
 */
class Device {
public:
    Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device();

    // Get the underlying MTL::Device
    MTL::Device* mtl_device() { return device_; }
    
    // Get device architecture name (e.g., "applegpu_g14s")
    const std::string& get_architecture() const { return arch_; }
    
    // Get architecture generation (e.g., 14 for M3)
    int get_architecture_gen() const { return arch_gen_; }
    
    // Check if this is Apple Silicon with unified memory
    bool has_unified_memory() const { return device_->hasUnifiedMemory(); }
    
    // Get maximum threadgroup size
    NS::UInteger max_threadgroup_size() const { 
        return device_->maxThreadsPerThreadgroup().width;
    }
    
    // Get recommended working set size for unified memory
    size_t recommended_working_set_size() const {
        return device_->recommendedMaxWorkingSetSize();
    }

    // Command queue management
    void new_queue(int index);
    MTL::CommandQueue* get_queue(int index);
    MTL::CommandBuffer* get_command_buffer(int index);
    void commit_command_buffer(int index);
    CommandEncoder& get_command_encoder(int index);
    void end_encoding(int index);

    // Library and kernel management
    MTL::Library* get_library(const std::string& name, const std::string& path = "");
    MTL::Library* get_library(const std::string& name,
                              const std::function<std::string(void)>& builder);
    
    MTL::ComputePipelineState* get_kernel(
        const std::string& base_name,
        MTL::Library* mtl_lib,
        const std::string& hash_name = "",
        const MTLFCList& func_consts = {});
    
    MTL::ComputePipelineState* get_kernel(
        const std::string& base_name,
        const std::string& hash_name = "",
        const MTLFCList& func_consts = {});

    // Allocate unified memory buffer
    MTL::Buffer* allocate_buffer(size_t size, MTL::ResourceOptions options = 
        MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked);
    
    // Create buffer from existing data (zero-copy for unified memory)
    MTL::Buffer* create_buffer_no_copy(void* ptr, size_t size);

private:
    DeviceStream& get_stream_(int index);
    MTL::Library* get_library_(const std::string& name);
    MTL::Library* build_library_(const std::string& source_string);
    MTL::Function* get_function_(const std::string& name, MTL::Library* mtl_lib);
    MTL::Function* get_function_(const std::string& name,
                                  const std::string& specialized_name,
                                  const MTLFCList& func_consts,
                                  MTL::Library* mtl_lib);
    MTL::ComputePipelineState* get_kernel_(const std::string& name,
                                            const MTL::Function* mtl_function);

    MTL::Device* device_;
    std::unordered_map<int, DeviceStream> stream_map_;
    
    std::shared_mutex kernel_mtx_;
    std::shared_mutex library_mtx_;
    std::unordered_map<std::string, MTL::Library*> library_map_;
    MTL::Library* default_library_;
    std::unordered_map<MTL::Library*, 
                       std::unordered_map<std::string, MTL::ComputePipelineState*>> 
        library_kernels_;
    
    std::string arch_;
    int arch_gen_;
    int max_ops_per_buffer_;
};

// Global device accessor
Device& device();

// RAII wrapper for NS::AutoreleasePool
std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool();

} // namespace Metal
} // namespace MetalFish

