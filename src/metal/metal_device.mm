/*
  MetalFish - GPU-accelerated chess engine - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  
  Metal device implementation using Objective-C++.
*/

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_device.h"
#include <iostream>
#include <unordered_map>

namespace MetalFish {

struct MetalDevice::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> commandQueue = nil;
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    
    bool initialize() {
        @autoreleasepool {
            // Get default Metal device
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                std::cerr << "MetalFish: No Metal device available" << std::endl;
                return false;
            }
            
            // Create command queue
            commandQueue = [device newCommandQueue];
            if (!commandQueue) {
                std::cerr << "MetalFish: Failed to create command queue" << std::endl;
                return false;
            }
            
            // Load shader library
            NSError* error = nil;
            NSString* shaderPath = [[NSBundle mainBundle] pathForResource:@"nnue" ofType:@"metallib"];
            
            if (shaderPath) {
                NSURL* shaderURL = [NSURL fileURLWithPath:shaderPath];
                library = [device newLibraryWithURL:shaderURL error:&error];
            }
            
            if (!library) {
                // Try to compile from source
                NSString* sourcePath = [[NSBundle mainBundle] pathForResource:@"nnue" ofType:@"metal"];
                if (sourcePath) {
                    NSString* source = [NSString stringWithContentsOfFile:sourcePath 
                                                                 encoding:NSUTF8StringEncoding 
                                                                    error:&error];
                    if (source) {
                        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
                        library = [device newLibraryWithSource:source options:options error:&error];
                    }
                }
            }
            
            if (!library) {
                // Compile from embedded source as fallback
                std::cerr << "MetalFish: Using embedded shader source" << std::endl;
                // This would contain the shader source inline
                // For now, we'll just note that GPU acceleration is unavailable
                return false;
            }
            
            std::cout << "MetalFish: Metal GPU acceleration initialized on " 
                      << [[device name] UTF8String] << std::endl;
            return true;
        }
    }
    
    id<MTLComputePipelineState> getPipeline(const std::string& name) {
        auto it = pipelines.find(name);
        if (it != pipelines.end()) {
            return it->second;
        }
        
        @autoreleasepool {
            NSString* nsName = [NSString stringWithUTF8String:name.c_str()];
            id<MTLFunction> function = [library newFunctionWithName:nsName];
            if (!function) {
                std::cerr << "MetalFish: Kernel not found: " << name << std::endl;
                return nil;
            }
            
            NSError* error = nil;
            id<MTLComputePipelineState> pipeline = 
                [device newComputePipelineStateWithFunction:function error:&error];
            
            if (!pipeline) {
                std::cerr << "MetalFish: Failed to create pipeline for " << name << std::endl;
                return nil;
            }
            
            pipelines[name] = pipeline;
            return pipeline;
        }
    }
};

MetalDevice& MetalDevice::instance() {
    static MetalDevice instance;
    return instance;
}

MetalDevice::MetalDevice() : pImpl(std::make_unique<Impl>()) {
    available = pImpl->initialize();
}

MetalDevice::~MetalDevice() = default;

void* MetalDevice::create_buffer(size_t size) {
    if (!available) return nullptr;
    
    @autoreleasepool {
        // Use shared storage mode for unified memory
        id<MTLBuffer> buffer = [pImpl->device newBufferWithLength:size 
                                                          options:MTLResourceStorageModeShared];
        if (!buffer) return nullptr;
        
        // Return the contents pointer (zero-copy access)
        return [buffer contents];
    }
}

void MetalDevice::release_buffer(void* buffer) {
    // In a real implementation, we'd track buffer objects
    // For now, rely on ARC
}

void MetalDevice::dispatch_compute(
    const std::string& kernel_name,
    const std::vector<void*>& buffers,
    const std::vector<size_t>& buffer_sizes,
    size_t grid_size_x,
    size_t grid_size_y,
    size_t grid_size_z
) {
    if (!available) return;
    
    @autoreleasepool {
        id<MTLComputePipelineState> pipeline = pImpl->getPipeline(kernel_name);
        if (!pipeline) return;
        
        id<MTLCommandBuffer> commandBuffer = [pImpl->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        
        // Set buffers
        for (size_t i = 0; i < buffers.size(); ++i) {
            // Create temporary buffer wrapper
            id<MTLBuffer> mtlBuffer = [pImpl->device newBufferWithBytesNoCopy:buffers[i]
                                                                       length:buffer_sizes[i]
                                                                      options:MTLResourceStorageModeShared
                                                                  deallocator:nil];
            [encoder setBuffer:mtlBuffer offset:0 atIndex:i];
        }
        
        // Calculate thread group size
        NSUInteger threadGroupSize = [pipeline maxTotalThreadsPerThreadgroup];
        if (threadGroupSize > 256) threadGroupSize = 256;
        
        MTLSize gridSize = MTLSizeMake(grid_size_x, grid_size_y, grid_size_z);
        MTLSize threadgroupSize = MTLSizeMake(
            std::min(grid_size_x, (size_t)threadGroupSize),
            1, 1
        );
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

void MetalDevice::synchronize() {
    // Already synchronous in current implementation
}

} // namespace MetalFish

#endif // __APPLE__
