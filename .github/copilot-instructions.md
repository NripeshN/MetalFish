# MetalFish Copilot Instructions

## Repository Overview

MetalFish is a high-performance UCI chess engine optimized for Apple Silicon with Metal GPU-accelerated NNUE evaluation. It implements three search algorithms: Alpha-Beta search (Stockfish-derived), MCTS (Monte Carlo Tree Search), and Hybrid MCTS-Alpha-Beta search. The codebase is ~106 C++ source files totaling a medium-sized project focused on chess AI with GPU acceleration.

**Languages & Frameworks:** C++20, Metal (macOS), CUDA (optional), Python (testing), CMake build system  
**Target Platforms:** macOS (primary with Metal), Linux (CPU/CUDA), Windows (CPU)  
**Repository Size:** ~106 source files, 3 main executables (metalfish, metalfish_tests, metalfish_gpu_bench)

## Critical Build Prerequisites

### NNUE Neural Network Files (REQUIRED)

**The build WILL FAIL without these files.** NNUE files are embedded into the binary during linking and are NOT optional:

```bash
cd src
curl -LO https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -LO https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
```

**Build Error if Missing:** Linker will fail with "Error: file not found: nn-c288c895ea92.nnue" during the final linking stage. Always download these files FIRST before building.

### Build Tools Required

- **CMake:** 3.20 or later
- **C++ Compiler:** Supporting C++20 (g++ 11+, clang 14+, MSVC 2022+)
- **macOS:** Xcode Command Line Tools (for Metal), macOS 12.0+
- **Linux/Windows:** Standard toolchains
- **Optional:** CUDA Toolkit 13.1.0+ (for CUDA builds)

## Build Instructions - EXACT STEPS

### macOS (Metal GPU - Recommended)

**Always run these commands in this exact order:**

```bash
# 1. FIRST: Download NNUE files (CRITICAL - build fails without these)
cd src
curl -LO https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -LO https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
cd ..

# 2. Configure CMake
mkdir -p build && cd build
cmake .. -DUSE_METAL=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# 3. Build with parallel jobs
cmake --build . --parallel

# 4. NNUE files are automatically copied to build directory
ls -la *.nnue  # Verify they're present

# 5. Run tests
./metalfish_tests
```

### Linux (CPU or CUDA)

```bash
# 1. FIRST: Download NNUE files (CRITICAL)
cd src
curl -LO https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -LO https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
cd ..

# 2. Configure for CPU build
mkdir -p build && cd build
cmake .. -DUSE_METAL=OFF -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# OR for CUDA build (if NVIDIA GPU and drivers available)
cmake .. -DUSE_METAL=OFF -DUSE_CUDA=ON -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build . --parallel

# 4. Run tests
./metalfish_tests
```

### Windows

```bash
# 1. FIRST: Download NNUE files in src/ directory
# 2. Configure
mkdir build && cd build
cmake .. -DUSE_METAL=OFF -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build . --config Release --parallel

# 4. NNUE files must be in build/Release/ for Windows
copy ..\src\*.nnue Release\

# 5. Run tests from build/Release/
cd Release
.\metalfish_tests.exe
```

## Testing - Complete Workflow

### C++ Unit Tests

```bash
cd build
./metalfish_tests  # Unix
# OR
./Release/metalfish_tests.exe  # Windows
```

### Python Perft Tests

**Requires:** Python 3.x, built MetalFish binary in build/

```bash
# Quick tests (recommended for CI/validation)
python3 tests/testing.py --quick

# Full perft test suite (slower)
python3 tests/testing.py

# Benchmark comparison (if Stockfish available)
python3 tests/testing.py --bench
```

**Execution Time:** `--quick` takes ~30-60 seconds, full suite can take 5+ minutes

### UCI Protocol Validation

```bash
cd build
printf "uci\nisready\nposition startpos\ngo depth 5\nquit\n" | ./metalfish
```

Expected output includes "uciok", "readyok", and "bestmove".

## Project Structure & Key Locations

```
metalfish/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Main CI/CD (builds macOS/Linux/Windows, runs tests)
│       └── elo-tournament.yml  # Tournament automation (macOS only)
├── CMakeLists.txt              # Main build configuration
├── .pre-commit-config.yaml     # Pre-commit hooks (clang-format, black, isort)
├── src/
│   ├── main.cpp               # Entry point for metalfish binary
│   ├── core/                  # Bitboard, position, move generation
│   ├── search/                # Alpha-Beta search (Stockfish-derived)
│   ├── eval/
│   │   └── nnue/              # Neural network evaluation
│   ├── mcts/                  # MCTS and hybrid search implementations
│   ├── gpu/                   # GPU acceleration framework
│   │   ├── metal/             # Metal backend (macOS)
│   │   ├── cuda/              # CUDA backend (Linux/Windows)
│   │   └── cpu_backend.cpp    # CPU fallback
│   ├── uci/                   # UCI protocol implementation
│   ├── syzygy/                # Tablebase probing
│   ├── nn-*.nnue              # NNUE files (download separately, NOT in git)
│   ├── benchmark_gpu.cpp      # GPU benchmark utility
│   ├── paper_benchmark.cpp    # Paper benchmarks
│   └── hybrid_benchmark.cpp   # Hybrid search benchmark
├── tests/
│   ├── test_*.cpp             # C++ unit tests (Catch2-style)
│   └── testing.py             # Python test runner (perft, UCI, benchmarks)
├── tools/
│   ├── elo_tournament.py      # Tournament automation script
│   └── metalfish_*_wrapper.sh # Engine mode wrappers (hybrid, mctsmt)
└── paper/                     # LaTeX academic paper
```

## CI/CD Validation Pipeline

The repository runs CI checks on every push/PR via `.github/workflows/ci.yml`:

### Build Matrix (All must pass)
- **macOS-latest (Metal):** Builds with Metal GPU, runs all tests
- **Ubuntu-latest (CPU):** CPU-only build, runs all tests
- **Windows-latest (CPU):** CPU-only build, runs all tests
- **Ubuntu-latest (CUDA):** Optional, compile-only (no GPU in CI)

### Test Stages (All must pass)
1. **CMake Configure:** Must complete without NNUE warnings if files downloaded
2. **Build:** All targets compile cleanly
3. **C++ Tests:** `./metalfish_tests` passes
4. **Perft Tests:** `python3 tests/testing.py --quick` passes
5. **UCI Protocol Test:** Basic UCI commands work

### Metal Shader Checks (macOS only)
- Metal shaders in `src/gpu/metal/kernels/*.metal` are syntax-checked with `xcrun metal`

## Common Build Issues & Workarounds

### Issue 1: Linker Error "file not found: nn-*.nnue"
**Cause:** NNUE files missing from src/ directory  
**Fix:** Download files BEFORE building (see Build Prerequisites above)  
**Prevention:** Always run download commands first in any build script

### Issue 2: CMake Warning "NNUE file not found"
**Cause:** Files not in src/ during configure  
**Impact:** Build will fail later during linking  
**Fix:** Download files, reconfigure: `rm -rf build && mkdir build && cd build && cmake ..`

### Issue 3: Metal Build Fails on Linux/Windows
**Cause:** Metal only works on macOS  
**Fix:** Use `-DUSE_METAL=OFF` for non-macOS platforms

### Issue 4: CUDA Tests Require GPU Drivers
**Note:** CUDA builds compile successfully but tests require actual NVIDIA GPU + drivers  
**CI Behavior:** CI compiles CUDA code but skips tests (no GPU available)

### Issue 5: Tests Fail with "MetalFish binary not found"
**Cause:** Build output location varies by platform (build/ vs build/Release/)  
**Fix:** Python test runner auto-detects location; ensure build completed successfully

## Making Code Changes

### Pre-commit Hooks
```bash
# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

Hooks run: `clang-format` (C++), `black` + `isort` (Python), `cmake-format`

### Typical Development Workflow
1. Make changes to source files
2. Rebuild: `cd build && cmake --build . --parallel`
3. Run relevant tests: `./metalfish_tests` or `python3 tests/testing.py --quick`
4. Test UCI manually: `printf "uci\nisready\nposition startpos\ngo depth 10\nquit\n" | ./metalfish`
5. Verify no Metal/CUDA shader issues if modified GPU code

### GPU Code Changes
- **Metal shaders:** `src/gpu/metal/kernels/*.metal` - verify with `xcrun metal -c <file>`
- **CUDA kernels:** `src/gpu/cuda/kernels/*.cu` - verify with `nvcc --syntax-only -x cu <file>`
- Run `./metalfish_gpu_bench` to validate GPU performance

## Important Notes

- **NNUE files are in .gitignore** - they are too large for Git and must be downloaded separately
- **Metal backend auto-downloads metal-cpp** during CMake configure if missing
- **Default build type is Release** - use `-DCMAKE_BUILD_TYPE=Debug` for debugging
- **LTO is enabled** by default for performance - increases link time but improves speed
- **Tests must run from build directory** - they expect NNUE files in current directory
- **Windows builds** put executables in build/Release/ - adjust paths accordingly

## Trust These Instructions

These instructions are comprehensive and tested. Only search for additional information if you encounter an issue not covered here, or if these instructions are incorrect. If NNUE download fails, that's a network issue - retry or use alternative download methods.
