/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  
  Based on Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers

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

#include <iostream>
#include <string>
#include <sstream>

#include "metal/device.h"
#include "core/types.h"

namespace MetalFish {

// Engine version
constexpr const char* VERSION = "1.0.0-dev";
constexpr const char* ENGINE_NAME = "MetalFish";

void print_info() {
    std::cout << ENGINE_NAME << " " << VERSION << std::endl;
    std::cout << "A GPU-accelerated UCI chess engine using Apple Metal" << std::endl;
    std::cout << "Copyright (C) 2025 Nripesh Niketan" << std::endl;
    std::cout << "Licensed under GPL-3.0" << std::endl;
    std::cout << std::endl;
    
    // Print Metal device info
    try {
        auto& dev = Metal::device();
        std::cout << "Metal Device: " << dev.get_architecture() << std::endl;
        std::cout << "Architecture Gen: " << dev.get_architecture_gen() << std::endl;
        std::cout << "Unified Memory: " << (dev.has_unified_memory() ? "Yes" : "No") << std::endl;
        std::cout << "Max Threadgroup Size: " << dev.max_threadgroup_size() << std::endl;
        std::cout << "Recommended Working Set: " << 
                     (dev.recommended_working_set_size() / (1024 * 1024)) << " MB" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error initializing Metal: " << e.what() << std::endl;
    }
}

void run_bench() {
    std::cout << "Running benchmark..." << std::endl;
    
    // TODO: Implement benchmark
    // - Perft at various depths
    // - Search speed test
    // - NNUE evaluation throughput
    
    std::cout << "Benchmark not yet implemented" << std::endl;
}

void uci_loop() {
    std::string line;
    std::string token;
    
    std::cout << "id name " << ENGINE_NAME << " " << VERSION << std::endl;
    std::cout << "id author Nripesh Niketan" << std::endl;
    
    // UCI options
    std::cout << "option name Hash type spin default 256 min 1 max 65536" << std::endl;
    std::cout << "option name Threads type spin default 1 min 1 max 1" << std::endl;
    std::cout << "option name UseGPU type check default true" << std::endl;
    std::cout << "option name GPUBatchSize type spin default 256 min 1 max 4096" << std::endl;
    
    std::cout << "uciok" << std::endl;
    
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        iss >> token;
        
        if (token == "quit") {
            break;
        }
        else if (token == "uci") {
            std::cout << "id name " << ENGINE_NAME << " " << VERSION << std::endl;
            std::cout << "id author Nripesh Niketan" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if (token == "isready") {
            std::cout << "readyok" << std::endl;
        }
        else if (token == "ucinewgame") {
            // TODO: Clear transposition table, reset state
        }
        else if (token == "position") {
            // TODO: Parse position
            std::string pos_type;
            iss >> pos_type;
            
            if (pos_type == "startpos") {
                // Set starting position
            }
            else if (pos_type == "fen") {
                // Parse FEN
            }
        }
        else if (token == "go") {
            // TODO: Parse go command and start search
            std::cout << "bestmove e2e4" << std::endl; // Placeholder
        }
        else if (token == "stop") {
            // TODO: Stop search
        }
        else if (token == "ponderhit") {
            // TODO: Handle ponder hit
        }
        else if (token == "setoption") {
            // TODO: Handle options
        }
        else if (token == "d" || token == "display") {
            // TODO: Display current position
        }
        else if (token == "bench") {
            run_bench();
        }
        else if (token == "eval") {
            // TODO: Show evaluation
        }
    }
}

} // namespace MetalFish

int main(int argc, char* argv[]) {
    // Handle command-line arguments
    if (argc > 1) {
        std::string arg = argv[1];
        
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: metalfish [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --help, -h     Show this help message" << std::endl;
            std::cout << "  --version, -v  Show version information" << std::endl;
            std::cout << "  --bench        Run benchmark" << std::endl;
            std::cout << "  --info         Show Metal device information" << std::endl;
            return 0;
        }
        else if (arg == "--version" || arg == "-v") {
            std::cout << MetalFish::ENGINE_NAME << " " << MetalFish::VERSION << std::endl;
            return 0;
        }
        else if (arg == "--bench") {
            MetalFish::print_info();
            MetalFish::run_bench();
            return 0;
        }
        else if (arg == "--info") {
            MetalFish::print_info();
            return 0;
        }
    }
    
    // Print startup info
    MetalFish::print_info();
    std::cout << std::endl;
    
    // Enter UCI loop
    MetalFish::uci_loop();
    
    return 0;
}

