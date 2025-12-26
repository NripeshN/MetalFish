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

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "metal/device.h"
#include <sstream>
#include <stdexcept>
#include <filesystem>
#include <mach-o/dyld.h>

namespace MetalFish {
namespace Metal {

namespace {

// Get the Metal language version based on OS
auto get_metal_version() {
    if (__builtin_available(macOS 15, iOS 18, *)) {
        return MTL::LanguageVersion3_2;
    } else if (__builtin_available(macOS 14, iOS 17, *)) {
        return MTL::LanguageVersion3_1;
    } else {
        return MTL::LanguageVersion3_0;
    }
}

// Load the default Metal device
MTL::Device* load_device() {
    auto device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        throw std::runtime_error("[MetalFish] Failed to create Metal device. "
                                "Metal is required for MetalFish.");
    }
    return device;
}

// Load a Metal library from file path
std::pair<MTL::Library*, NS::Error*> load_library_from_path(
    MTL::Device* device,
    const char* path) {
    auto library = NS::String::string(path, NS::UTF8StringEncoding);
    NS::Error* error = nullptr;
    auto lib = device->newLibrary(library, &error);
    return {lib, error};
}

// Get current binary directory using mach-o APIs
std::filesystem::path current_binary_dir() {
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        return std::filesystem::path(path).parent_path();
    }
    return std::filesystem::current_path();
}

} // namespace

// CommandEncoder implementation
CommandEncoder::CommandEncoder(DeviceStream& stream) : stream_(stream) {
    enc_ = stream_.buffer->computeCommandEncoder(MTL::DispatchTypeConcurrent);
    if (!enc_) {
        throw std::runtime_error("[MetalFish] Failed to create compute command encoder");
    }
    enc_->retain();
}

CommandEncoder::~CommandEncoder() {
    enc_->endEncoding();
    enc_->release();
}

void CommandEncoder::set_buffer(const MTL::Buffer* buf, int idx, int64_t offset) {
    enc_->setBuffer(buf, offset, idx);
}

void CommandEncoder::dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims) {
    if (needs_barrier_) {
        enc_->memoryBarrier(MTL::BarrierScopeBuffers);
        needs_barrier_ = false;
    }
    stream_.buffer_ops++;
    enc_->dispatchThreadgroups(grid_dims, group_dims);
}

void CommandEncoder::dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims) {
    if (needs_barrier_) {
        enc_->memoryBarrier(MTL::BarrierScopeBuffers);
        needs_barrier_ = false;
    }
    stream_.buffer_ops++;
    enc_->dispatchThreads(grid_dims, group_dims);
}

void CommandEncoder::set_compute_pipeline_state(MTL::ComputePipelineState* kernel) {
    enc_->setComputePipelineState(kernel);
}

void CommandEncoder::set_threadgroup_memory_length(size_t length, int idx) {
    enc_->setThreadgroupMemoryLength(length, idx);
}

void CommandEncoder::barrier() {
    enc_->memoryBarrier(MTL::BarrierScopeBuffers);
}

// DeviceStream implementation
DeviceStream::DeviceStream(MTL::CommandQueue* q) : queue(q) {}

DeviceStream::~DeviceStream() {
    if (queue) {
        queue->release();
    }
    if (buffer) {
        buffer->release();
    }
}

// Device implementation
Device::Device() {
    auto pool = new_scoped_memory_pool();
    
    device_ = load_device();
    
    // Extract architecture info
    arch_ = std::string(device_->architecture()->name()->utf8String());
    
    // Parse architecture generation (e.g., "applegpu_g14s" -> 14)
    size_t pos = arch_.find_last_of("g");
    if (pos != std::string::npos && pos + 2 < arch_.size()) {
        int tens = 0, ones = 0;
        if (std::isdigit(arch_[pos + 1])) {
            tens = arch_[pos + 1] - '0';
        }
        if (pos + 2 < arch_.size() && std::isdigit(arch_[pos + 2])) {
            ones = arch_[pos + 2] - '0';
            arch_gen_ = tens * 10 + ones;
        } else {
            arch_gen_ = tens;
        }
    } else {
        arch_gen_ = 13; // Default to M1/M2 generation
    }
    
    // Set max operations per buffer based on device
    if (arch_.find('s') != std::string::npos) {
        // Max chip
        max_ops_per_buffer_ = 50;
    } else if (arch_.find('d') != std::string::npos) {
        // Ultra chip
        max_ops_per_buffer_ = 50;
    } else {
        // Base/Pro chip
        max_ops_per_buffer_ = 40;
    }
    
    // Load default metallib
    std::string metallib_path = METAL_PATH;
    auto [lib, error] = load_library_from_path(device_, metallib_path.c_str());
    if (lib) {
        default_library_ = lib;
    } else {
        // Try current directory
        auto cwd_path = std::filesystem::current_path() / "metalfish.metallib";
        std::tie(lib, error) = load_library_from_path(device_, cwd_path.c_str());
        if (lib) {
            default_library_ = lib;
        } else {
            // Library will be loaded later or built from source
            default_library_ = nullptr;
        }
    }
}

Device::~Device() {
    auto pool = new_scoped_memory_pool();
    
    // Release all kernels and libraries
    for (auto& [lib, kernel_map] : library_kernels_) {
        for (auto& [_, kernel] : kernel_map) {
            kernel->release();
        }
        lib->release();
    }
    
    // Release default library
    if (default_library_) {
        default_library_->release();
    }
    
    // Clear streams
    stream_map_.clear();
    
    // Release device
    if (device_) {
        device_->release();
    }
}

void Device::new_queue(int index) {
    auto pool = new_scoped_memory_pool();
    auto queue = device_->newCommandQueue();
    if (!queue) {
        throw std::runtime_error("[MetalFish] Failed to create command queue");
    }
    stream_map_.emplace(index, queue);
}

MTL::CommandQueue* Device::get_queue(int index) {
    return get_stream_(index).queue;
}

DeviceStream& Device::get_stream_(int index) {
    auto it = stream_map_.find(index);
    if (it == stream_map_.end()) {
        throw std::runtime_error("[MetalFish] Stream not found: " + std::to_string(index));
    }
    return it->second;
}

MTL::CommandBuffer* Device::get_command_buffer(int index) {
    auto& stream = get_stream_(index);
    if (stream.buffer == nullptr) {
        stream.buffer = stream.queue->commandBufferWithUnretainedReferences();
        if (!stream.buffer) {
            throw std::runtime_error("[MetalFish] Failed to create command buffer");
        }
        stream.buffer->retain();
    }
    return stream.buffer;
}

void Device::commit_command_buffer(int index) {
    auto& stream = get_stream_(index);
    if (stream.buffer) {
        stream.buffer->commit();
        stream.buffer->release();
        stream.buffer = nullptr;
        stream.buffer_ops = 0;
    }
}

CommandEncoder& Device::get_command_encoder(int index) {
    auto& stream = get_stream_(index);
    if (stream.encoder == nullptr) {
        if (stream.buffer == nullptr) {
            get_command_buffer(index);
        }
        stream.encoder = std::make_unique<CommandEncoder>(stream);
    }
    return *stream.encoder;
}

void Device::end_encoding(int index) {
    auto& stream = get_stream_(index);
    stream.encoder = nullptr;
}

MTL::Library* Device::get_library(const std::string& name, const std::string& path) {
    {
        std::shared_lock lock(library_mtx_);
        auto it = library_map_.find(name);
        if (it != library_map_.end()) {
            return it->second;
        }
    }
    
    std::unique_lock lock(library_mtx_);
    
    // Double-check after acquiring write lock
    auto it = library_map_.find(name);
    if (it != library_map_.end()) {
        return it->second;
    }
    
    std::string full_path = path.empty() ? name + ".metallib" : path;
    auto [lib, error] = load_library_from_path(device_, full_path.c_str());
    
    if (!lib) {
        std::ostringstream msg;
        msg << "[MetalFish] Failed to load Metal library: " << name;
        if (error) {
            msg << " - " << error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(msg.str());
    }
    
    library_map_[name] = lib;
    return lib;
}

MTL::Library* Device::get_library(const std::string& name,
                                   const std::function<std::string(void)>& builder) {
    {
        std::shared_lock lock(library_mtx_);
        auto it = library_map_.find(name);
        if (it != library_map_.end()) {
            return it->second;
        }
    }
    
    std::unique_lock lock(library_mtx_);
    
    auto it = library_map_.find(name);
    if (it != library_map_.end()) {
        return it->second;
    }
    
    auto lib = build_library_(builder());
    library_map_[name] = lib;
    return lib;
}

MTL::Library* Device::build_library_(const std::string& source_string) {
    auto pool = new_scoped_memory_pool();
    
    auto ns_code = NS::String::string(source_string.c_str(), NS::ASCIIStringEncoding);
    
    NS::Error* error = nullptr;
    auto options = MTL::CompileOptions::alloc()->init();
    options->setFastMathEnabled(true);
    options->setLanguageVersion(get_metal_version());
    
    auto mtl_lib = device_->newLibrary(ns_code, options, &error);
    options->release();
    
    if (!mtl_lib) {
        std::ostringstream msg;
        msg << "[MetalFish] Failed to compile Metal library";
        if (error) {
            msg << ": " << error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(msg.str());
    }
    
    return mtl_lib;
}

MTL::Function* Device::get_function_(const std::string& name, MTL::Library* mtl_lib) {
    auto ns_name = NS::String::string(name.c_str(), NS::ASCIIStringEncoding);
    return mtl_lib->newFunction(ns_name);
}

MTL::Function* Device::get_function_(const std::string& name,
                                      const std::string& specialized_name,
                                      const MTLFCList& func_consts,
                                      MTL::Library* mtl_lib) {
    if (func_consts.empty() && specialized_name == name) {
        return get_function_(name, mtl_lib);
    }
    
    auto mtl_func_consts = MTL::FunctionConstantValues::alloc()->init();
    
    for (const auto& [value, type, index] : func_consts) {
        mtl_func_consts->setConstantValue(value, type, index);
    }
    
    auto desc = MTL::FunctionDescriptor::functionDescriptor();
    desc->setName(NS::String::string(name.c_str(), NS::ASCIIStringEncoding));
    desc->setSpecializedName(NS::String::string(specialized_name.c_str(), NS::ASCIIStringEncoding));
    desc->setConstantValues(mtl_func_consts);
    
    NS::Error* error = nullptr;
    auto mtl_function = mtl_lib->newFunction(desc, &error);
    
    mtl_func_consts->release();
    
    if (!mtl_function) {
        std::ostringstream msg;
        msg << "[MetalFish] Failed to load function: " << name;
        if (error) {
            msg << " - " << error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(msg.str());
    }
    
    return mtl_function;
}

MTL::ComputePipelineState* Device::get_kernel_(const std::string& name,
                                                const MTL::Function* mtl_function) {
    NS::Error* error = nullptr;
    auto kernel = device_->newComputePipelineState(mtl_function, &error);
    
    if (!kernel) {
        std::ostringstream msg;
        msg << "[MetalFish] Failed to create compute pipeline: " << name;
        if (error) {
            msg << " - " << error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(msg.str());
    }
    
    return kernel;
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    MTL::Library* mtl_lib,
    const std::string& hash_name,
    const MTLFCList& func_consts) {
    
    const auto& kname = hash_name.empty() ? base_name : hash_name;
    
    {
        std::shared_lock lock(kernel_mtx_);
        auto lib_it = library_kernels_.find(mtl_lib);
        if (lib_it != library_kernels_.end()) {
            auto kernel_it = lib_it->second.find(kname);
            if (kernel_it != lib_it->second.end()) {
                return kernel_it->second;
            }
        }
    }
    
    std::unique_lock lock(kernel_mtx_);
    
    // Double-check
    auto& kernel_map = library_kernels_[mtl_lib];
    auto it = kernel_map.find(kname);
    if (it != kernel_map.end()) {
        return it->second;
    }
    
    auto pool = new_scoped_memory_pool();
    
    auto mtl_function = get_function_(base_name, kname, func_consts, mtl_lib);
    auto kernel = get_kernel_(kname, mtl_function);
    mtl_function->release();
    
    kernel_map[kname] = kernel;
    return kernel;
}

MTL::ComputePipelineState* Device::get_kernel(
    const std::string& base_name,
    const std::string& hash_name,
    const MTLFCList& func_consts) {
    return get_kernel(base_name, default_library_, hash_name, func_consts);
}

MTL::Buffer* Device::allocate_buffer(size_t size, MTL::ResourceOptions options) {
    return device_->newBuffer(size, options);
}

MTL::Buffer* Device::create_buffer_no_copy(void* ptr, size_t size) {
    // Create buffer that uses existing memory (zero-copy)
    return device_->newBuffer(ptr, size, 
        MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked,
        nullptr);
}

// Global device accessor
Device& device() {
    static Device metal_device;
    return metal_device;
}

// Scoped memory pool
std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool() {
    auto dtor = [](void* ptr) {
        static_cast<NS::AutoreleasePool*>(ptr)->release();
    };
    return std::unique_ptr<void, std::function<void(void*)>>(
        NS::AutoreleasePool::alloc()->init(), dtor);
}

} // namespace Metal
} // namespace MetalFish

