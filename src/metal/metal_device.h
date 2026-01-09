/*
  MetalFish - GPU-accelerated chess engine - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Metal device wrapper for GPU operations.
*/

#ifndef METAL_DEVICE_H_INCLUDED
#define METAL_DEVICE_H_INCLUDED

#ifdef __APPLE__

#include <memory>
#include <string>
#include <vector>

namespace MetalFish {

class MetalDevice {
public:
  static MetalDevice &instance();

  bool is_available() const { return available; }

  // Create a buffer in unified memory (zero-copy)
  void *create_buffer(size_t size);
  void release_buffer(void *buffer);

  // Execute a compute shader
  void dispatch_compute(const std::string &kernel_name,
                        const std::vector<void *> &buffers,
                        const std::vector<size_t> &buffer_sizes,
                        size_t grid_size_x, size_t grid_size_y = 1,
                        size_t grid_size_z = 1);

  // Wait for all GPU operations to complete
  void synchronize();

private:
  MetalDevice();
  ~MetalDevice();
  MetalDevice(const MetalDevice &) = delete;
  MetalDevice &operator=(const MetalDevice &) = delete;

  struct Impl;
  std::unique_ptr<Impl> pImpl;
  bool available = false;
};

} // namespace MetalFish

#endif // __APPLE__

#endif // METAL_DEVICE_H_INCLUDED
