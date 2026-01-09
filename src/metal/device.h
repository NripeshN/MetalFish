/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

*/

#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace MetalFish {
namespace Metal {

struct DeviceStream;

// Command encoder for dispatching compute kernels
struct CommandEncoder {
  explicit CommandEncoder(DeviceStream &stream);
  ~CommandEncoder();

  void set_buffer(const MTL::Buffer *buf, int idx, int64_t offset = 0);
  void set_bytes(const void *data, size_t size, int idx);
  void set_compute_pipeline_state(MTL::ComputePipelineState *kernel);
  void dispatch_threadgroups(MTL::Size grid_dims, MTL::Size group_dims);
  void dispatch_threads(MTL::Size grid_dims, MTL::Size group_dims);
  void barrier();

private:
  DeviceStream &stream_;
  MTL::ComputeCommandEncoder *enc_;
};

// Stream for command buffer management
struct DeviceStream {
  DeviceStream(MTL::CommandQueue *q)
      : queue(q), buffer(nullptr), encoder(nullptr) {}
  ~DeviceStream() {
    if (buffer)
      buffer->release();
  }

  // Move only
  DeviceStream(DeviceStream &&other) noexcept
      : queue(other.queue), buffer(other.buffer),
        encoder(std::move(other.encoder)) {
    other.queue = nullptr;
    other.buffer = nullptr;
  }
  DeviceStream &operator=(DeviceStream &&other) noexcept {
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

  // Non-copyable
  DeviceStream(const DeviceStream &) = delete;
  DeviceStream &operator=(const DeviceStream &) = delete;

  MTL::CommandQueue *queue;
  MTL::CommandBuffer *buffer;
  std::unique_ptr<CommandEncoder> encoder;
};

// Main Metal device wrapper
class Device {
public:
  Device();
  ~Device();

  // Disable copy
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;

  MTL::Device *mtl_device();

  MTL::CommandQueue *get_queue(int index = 0);
  MTL::CommandBuffer *get_command_buffer(int index = 0);
  CommandEncoder &get_command_encoder(int index = 0);
  void commit_command_buffer(int index = 0);
  void end_encoding(int index = 0);

  MTL::Library *get_library(const std::string &name,
                            const std::string &path = "");
  MTL::ComputePipelineState *get_kernel(const std::string &name,
                                        MTL::Library *mtl_lib = nullptr);

  const std::string &get_architecture() const;
  const std::string &get_architecture_gen() const;

private:
  MTL::Device *device_;
  std::unordered_map<int32_t, DeviceStream> stream_map_;
  std::unordered_map<std::string, MTL::Library *> libraries_;
  std::unordered_map<std::string, MTL::ComputePipelineState *> kernels_;
  std::string architecture_;
  std::string architectureGen_;
};

// Global device accessor
Device &get_device();

} // namespace Metal
} // namespace MetalFish
