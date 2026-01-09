/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

*/

#ifdef USE_METAL

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "device.h"
#include <iostream>
#include <stdexcept>

namespace MetalFish {
namespace Metal {

// DeviceStream implementation
DeviceStream::DeviceStream(MTL::CommandQueue *q)
    : queue(q), buffer(nullptr), encoder(nullptr) {}

DeviceStream::~DeviceStream() {
  if (buffer)
    buffer->release();
}

DeviceStream::DeviceStream(DeviceStream &&other) noexcept
    : queue(other.queue), buffer(other.buffer),
      encoder(std::move(other.encoder)) {
  other.queue = nullptr;
  other.buffer = nullptr;
}

DeviceStream &DeviceStream::operator=(DeviceStream &&other) noexcept {
  if (this != &other) {
    if (buffer)
      buffer->release();
    queue = other.queue;
    buffer = other.buffer;
    encoder = std::move(other.encoder);
    other.queue = nullptr;
    other.buffer = nullptr;
  }
  return *this;
}

// CommandEncoder implementation
CommandEncoder::CommandEncoder(DeviceStream &stream)
    : stream_(stream), enc_(nullptr) {
  if (!stream_.buffer) {
    stream_.buffer = stream_.queue->commandBuffer();
    stream_.buffer->retain();
  }

  enc_ = stream_.buffer->computeCommandEncoder();
  if (!enc_) {
    throw std::runtime_error("Failed to create compute command encoder");
  }
  enc_->retain();
}

CommandEncoder::~CommandEncoder() {
  if (enc_) {
    enc_->endEncoding();
    enc_->release();
  }
}

void CommandEncoder::set_buffer(const MTL::Buffer *buf, int idx,
                                int64_t offset) {
  if (enc_ && buf) {
    enc_->setBuffer(const_cast<MTL::Buffer *>(buf), offset, idx);
  }
}

void CommandEncoder::set_compute_pipeline_state(
    MTL::ComputePipelineState *kernel) {
  if (enc_ && kernel) {
    enc_->setComputePipelineState(kernel);
  }
}

void CommandEncoder::dispatch_threadgroups(MTL::Size grid_dims,
                                           MTL::Size group_dims) {
  if (enc_) {
    enc_->dispatchThreadgroups(grid_dims, group_dims);
  }
}

void CommandEncoder::dispatch_threads(MTL::Size grid_dims,
                                      MTL::Size group_dims) {
  if (enc_) {
    enc_->dispatchThreads(grid_dims, group_dims);
  }
}

void CommandEncoder::barrier() {
  if (enc_) {
    enc_->memoryBarrier(MTL::BarrierScopeBuffers);
  }
}

void CommandEncoder::set_bytes(const void *data, size_t size, int idx) {
  if (enc_ && data) {
    enc_->setBytes(data, size, idx);
  }
}

// Device implementation
Device::Device() {
  // Get the default Metal device
  device_ = MTL::CreateSystemDefaultDevice();
  if (!device_) {
    throw std::runtime_error("Failed to create Metal device");
  }

  // Create default command queue
  MTL::CommandQueue *queue = device_->newCommandQueue();
  if (!queue) {
    throw std::runtime_error("Failed to create command queue");
  }

  stream_map_.try_emplace(0, queue);

  // Get device info
  architecture_ = device_->name()->utf8String();

  // Determine architecture generation
  if (architecture_.find("M1") != std::string::npos) {
    architectureGen_ = "Apple Silicon M1";
  } else if (architecture_.find("M2") != std::string::npos) {
    architectureGen_ = "Apple Silicon M2";
  } else if (architecture_.find("M3") != std::string::npos) {
    architectureGen_ = "Apple Silicon M3";
  } else if (architecture_.find("M4") != std::string::npos) {
    architectureGen_ = "Apple Silicon M4";
  } else {
    architectureGen_ = "Unknown";
  }

  std::cout << "[MetalFish] Initialized Metal device: " << architecture_
            << std::endl;
  std::cout << "[MetalFish] Unified memory: "
            << (device_->hasUnifiedMemory() ? "Yes" : "No") << std::endl;
  std::cout << "[MetalFish] Max threadgroup memory: "
            << device_->maxThreadgroupMemoryLength() << " bytes" << std::endl;
}

Device::~Device() {
  for (auto &[name, lib] : libraries_) {
    if (lib)
      lib->release();
  }
  for (auto &[name, kernel] : kernels_) {
    if (kernel)
      kernel->release();
  }
  for (auto &[idx, stream] : stream_map_) {
    if (stream.queue)
      stream.queue->release();
  }
  if (device_)
    device_->release();
}

MTL::Device *Device::mtl_device() { return device_; }

bool Device::has_unified_memory() const {
  return device_ ? device_->hasUnifiedMemory() : false;
}

MTL::CommandQueue *Device::get_queue(int index) {
  auto it = stream_map_.find(index);
  if (it != stream_map_.end()) {
    return it->second.queue;
  }
  return nullptr;
}

MTL::CommandBuffer *Device::get_command_buffer(int index) {
  auto it = stream_map_.find(index);
  if (it == stream_map_.end()) {
    return nullptr;
  }

  if (!it->second.buffer) {
    it->second.buffer = it->second.queue->commandBuffer();
    it->second.buffer->retain();
  }

  return it->second.buffer;
}

CommandEncoder &Device::get_command_encoder(int index) {
  auto it = stream_map_.find(index);
  if (it == stream_map_.end()) {
    throw std::runtime_error("Invalid stream index");
  }

  if (!it->second.encoder) {
    it->second.encoder = std::make_unique<CommandEncoder>(it->second);
  }

  return *it->second.encoder;
}

void Device::commit_command_buffer(int index) {
  auto it = stream_map_.find(index);
  if (it == stream_map_.end()) {
    return;
  }

  // End encoding if active
  it->second.encoder.reset();

  if (it->second.buffer) {
    it->second.buffer->commit();
    it->second.buffer->waitUntilCompleted();
    it->second.buffer->release();
    it->second.buffer = nullptr;
  }
}

void Device::end_encoding(int index) {
  auto it = stream_map_.find(index);
  if (it != stream_map_.end()) {
    it->second.encoder.reset();
  }
}

MTL::Library *Device::get_library(const std::string &name,
                                  const std::string &path) {
  // Check cache
  auto it = libraries_.find(name);
  if (it != libraries_.end()) {
    return it->second;
  }

  MTL::Library *library = nullptr;
  NS::Error *error = nullptr;

  if (!path.empty()) {
    // Load from file
    NS::String *nsPath =
        NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    NS::URL *url = NS::URL::fileURLWithPath(nsPath);
    library = device_->newLibrary(url, &error);
  } else {
    // Try to load default library
    library = device_->newDefaultLibrary();
  }

  if (error || !library) {
    std::cerr << "[MetalFish] Failed to load library: " << name << std::endl;
    return nullptr;
  }

  libraries_[name] = library;
  return library;
}

MTL::ComputePipelineState *Device::get_kernel(const std::string &name,
                                              MTL::Library *mtl_lib) {
  // Check cache
  auto it = kernels_.find(name);
  if (it != kernels_.end()) {
    return it->second;
  }

  if (!mtl_lib) {
    mtl_lib = device_->newDefaultLibrary();
    if (!mtl_lib) {
      std::cerr << "[MetalFish] No library available for kernel: " << name
                << std::endl;
      return nullptr;
    }
  }

  NS::String *nsName = NS::String::string(name.c_str(), NS::UTF8StringEncoding);
  MTL::Function *function = mtl_lib->newFunction(nsName);

  if (!function) {
    std::cerr << "[MetalFish] Function not found: " << name << std::endl;
    return nullptr;
  }

  NS::Error *error = nullptr;
  MTL::ComputePipelineState *kernel =
      device_->newComputePipelineState(function, &error);
  function->release();

  if (error || !kernel) {
    std::cerr << "[MetalFish] Failed to create pipeline for: " << name
              << std::endl;
    return nullptr;
  }

  kernels_[name] = kernel;
  return kernel;
}

const std::string &Device::get_architecture() const { return architecture_; }

const std::string &Device::get_architecture_gen() const {
  return architectureGen_;
}

// Global device accessor
Device &get_device() {
  static Device device;
  return device;
}

} // namespace Metal
} // namespace MetalFish

#endif // USE_METAL
