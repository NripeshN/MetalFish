/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  
  Based on Stockfish, Copyright (C) 2004-2025 The Stockfish developers

  MetalFish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
*/

#include <iostream>
#include <string>

#include "core/types.h"
#include "core/bitboard.h"
#include "core/position.h"
#include "uci/uci.h"
#include "metal/device.h"
#include "metal/allocator.h"

int main(int argc, char* argv[]) {
    // Disable sync for faster I/O
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    // Print banner
    std::cout << "MetalFish 1.0.0 - GPU-accelerated UCI Chess Engine\n";
    std::cout << "Copyright (C) 2025 Nripesh Niketan\n";
    std::cout << "Licensed under GPL-3.0\n";
    std::cout << "Based on Stockfish by The Stockfish developers\n\n";
    
    // Initialize Metal device
    try {
        MetalFish::Metal::Device& device = MetalFish::Metal::get_device();
        std::cout << "GPU: " << device.get_architecture() << "\n";
        std::cout << "Architecture: " << device.get_architecture_gen() << "\n";
        std::cout << "Unified Memory: " << (device.mtl_device()->hasUnifiedMemory() ? "Yes" : "No") << "\n";
        std::cout << "Max Threadgroup Size: " << device.mtl_device()->maxThreadsPerThreadgroup().width << "\n";
        std::cout << "Recommended Working Set: " 
                  << device.mtl_device()->recommendedMaxWorkingSetSize() / (1024 * 1024) << " MB\n\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Warning: Failed to initialize Metal device: " << e.what() << "\n";
        std::cerr << "Continuing with CPU-only mode.\n\n";
    }
    
    // Initialize bitboards
    MetalFish::init_bitboards();
    
    // Initialize position
    MetalFish::Position::init();
    
    // Start UCI loop
    MetalFish::UCI::loop(argc, argv);
    
    return 0;
}
