# MetalFish Copilot Instructions

## Repository Overview

**MetalFish** is a high-performance UCI chess engine optimized for Apple Silicon with Metal GPU acceleration, CUDA support for NVIDIA GPUs, and CPU fallback. The project implements three search modes: traditional Alpha-Beta search with NNUE evaluation, Monte Carlo Tree Search (MCTS) with GPU batching, and a parallel hybrid approach combining both.

**Key Stats:**
- Language: C++20 with Objective-C++ (macOS Metal), CUDA (NVIDIA GPU)
- Build System: CMake 3.20+ with Ninja (recommended)
- Target Platforms: macOS (Apple Silicon), Linux, Windows
- Test Framework: Custom C++ test suite + Python perft tests
- Size: Medium (~50K lines of code across core, search, eval, mcts, gpu, uci modules)

## Critical Build Prerequisites

**ALWAYS download NNUE network files BEFORE building:**
```bash
cd src
curl -L -O https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -L -O https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
```

**Build will fail** if NNUE files are missing - the linker embeds these files into the binary and requires them at link time.

## Build Instructions

### Standard Build Process (Verified Working)

**Clean Build (Recommended):**
```bash
# Step 1: Download NNUE files if not present
cd src
[ ! -f nn-c288c895ea92.nnue ] && curl -L -O https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
[ ! -f nn-37f18f62d772.nnue ] && curl -L -O https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
cd ..

# Step 2: Configure
mkdir -p build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=OFF -DUSE_CUDA=OFF -DBUILD_TESTS=ON

# Step 3: Build (takes ~90-120 seconds on typical CI runner)
ninja metalfish
# Or use: cmake --build . --config Release -j$(nproc)
```

**Platform-Specific Build Options:**
- **macOS with Metal GPU:** `-DUSE_METAL=ON` (default on macOS, downloads metal-cpp automatically)
- **Linux/Windows with CUDA:** `-DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=$(which nvcc)` (requires CUDA toolkit)
- **CPU fallback:** `-DUSE_METAL=OFF -DUSE_CUDA=OFF` (default on non-macOS, works everywhere)

**Important Notes:**
- CMake automatically downloads metal-cpp headers on macOS if missing
- Metal shaders compile automatically via xcrun on macOS
- Build uses LTO (`-flto`) which shows harmless warnings: "lto-wrapper: warning: using serial compilation"
- NNUE files are 104MB + 3.4MB and required at link time

### Build Troubleshooting

**Common Issues:**

1. **Linker error: "file not found: nn-*.nnue"**
   - **Cause:** NNUE files missing from src/ directory
   - **Fix:** Download NNUE files as shown above BEFORE running cmake --build

2. **"metal-cpp headers not found"**
   - **Cause:** Metal-cpp not downloaded on macOS
   - **Fix:** CMake automatically downloads it, but if it fails, run cmake again or manually download from https://developer.apple.com/metal/cpp/

3. **CUDA build fails on CI**
   - **Cause:** CUDA runtime requires GPU drivers (not available on GitHub Actions)
   - **Fix:** CUDA builds compile but don't run tests in CI (expected behavior)

## Testing

### Running Tests (Always run in this order)

**1. C++ Unit Tests:**
```bash
cd build
./metalfish_tests          # Unix (macOS/Linux)
# OR
./Release/metalfish_tests.exe  # Windows
```
- Expected: "ALL TESTS PASSED!" with 8 test suites (core, search, mcts, hybrid, gpu, metal, cuda, gpu_nnue)
- GPU tests skip if no GPU available (normal behavior)
- Takes ~5-10 seconds

**2. Python Perft Tests:**
```bash
python3 tests/testing.py --quick
```
- Expected: 30 passed perft tests + 9 passed UCI tests
- Takes ~15-20 seconds
- Tests move generation correctness via perft

**3. UCI Protocol Test:**
```bash
cd build
printf "uci\nisready\nposition startpos\ngo depth 5\nquit\n" | ./metalfish
```
- Expected: Engine responds with "uciok", "readyok", then "bestmove"
- Takes ~3-5 seconds

### Test Requirements
- C++ tests: None (self-contained)
- Python tests: `python3` and `chess` library (install via `pip3 install chess`)

## Project Structure

**Root Files:**
- `CMakeLists.txt` - Main build configuration (514 lines, comprehensive)
- `.pre-commit-config.yaml` - Pre-commit hooks (clang-format, black, isort, cmake-format)
- `.gitignore` - Excludes build/, external/, reference/, *.nnue files

**Source Layout (`src/`):**
```
src/
├── main.cpp                    # Entry point
├── core/                       # Chess fundamentals
│   ├── bitboard.{h,cpp}        # Bitboard ops, magic bitboards
│   ├── position.{h,cpp}        # Board state, FEN parsing
│   ├── movegen.{h,cpp}         # Legal move generation
│   ├── types.h                 # Core types (Square, Piece, Move, etc.)
│   └── misc.{h,cpp}            # Utilities
├── search/                     # Alpha-Beta search
│   ├── search.{h,cpp}          # Main search (PVS, aspiration, pruning)
│   ├── movepick.{h,cpp}        # Move ordering
│   ├── thread.{h,cpp}          # Thread pool
│   └── tt.{h,cpp}              # Transposition table
├── eval/                       # NNUE evaluation
│   └── nnue/                   # Neural network
│       ├── network.cpp         # NNUE architecture
│       └── features/           # Feature extractors
├── mcts/                       # MCTS implementation
│   ├── thread_safe_mcts.{h,cpp}        # Pure MCTS
│   ├── parallel_hybrid_search.{h,cpp}  # Parallel MCTS+AB
│   ├── apple_silicon_mcts.{h,cpp}      # Apple optimizations
│   └── ab_integration.{h,cpp}          # Alpha-Beta bridge
├── gpu/                        # GPU acceleration
│   ├── gpu_nnue_integration.{h,cpp}    # GPU NNUE manager
│   ├── gpu_mcts_backend.{h,cpp}        # MCTS GPU backend
│   ├── metal/                  # Metal implementation (macOS)
│   │   ├── metal_backend.mm    # Metal backend
│   │   └── kernels/nnue.metal  # Compute shaders
│   ├── cuda/                   # CUDA implementation (NVIDIA)
│   │   └── kernels/*.cu        # CUDA kernels
│   └── cpu_backend.cpp         # CPU fallback
├── uci/                        # UCI protocol
│   ├── uci.cpp                 # UCI command parsing
│   └── engine.cpp              # Engine interface
└── syzygy/                     # Tablebase support
    └── tbprobe.cpp
```

**Tests (`tests/`):**
- `test_main.cpp` - Test runner
- `test_core.cpp` - Core module tests
- `test_search_module.cpp` - Search tests
- `test_mcts_module.cpp` - MCTS tests
- `test_gpu_nnue.cpp` - GPU NNUE tests
- `testing.py` - Perft + UCI protocol tests

**Tools (`tools/`):**
- `elo_tournament.py` - Automated Elo tournament system (uses cutechess-cli)
- `engines_config.json` - Engine configurations

## CI/CD Pipeline

**GitHub Actions Workflows:**

1. **`.github/workflows/ci.yml`** - Main CI pipeline
   - Builds on: Ubuntu (CPU), Windows (CPU), macOS (Metal)
   - Optional CUDA build job (compile-only, no GPU tests)
   - Runs: C++ tests, perft tests, UCI protocol test
   - Artifacts: Executables for each platform
   - Release job: Creates GitHub releases on version tags

2. **`.github/workflows/elo-tournament.yml`** - Elo rating tournament
   - Builds reference engines (Stockfish, Patricia, Berserk)
   - Runs matches using cutechess-cli
   - Only on PRs or manual trigger

**CI Build Sequence (from ci.yml):**
```bash
# 1. Download NNUE files
cd src
curl -L --retry 3 --retry-delay 2 -O https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -L --retry 3 --retry-delay 2 -O https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue

# 2. Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=[ON/OFF] -DBUILD_TESTS=ON

# 3. Build (Unix)
cmake --build build --config Release -j$(nproc)
# OR (Windows)
cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS

# 4. Copy NNUE to build output (Windows needs this)
cp src/*.nnue build/Release/ 2>/dev/null || true  # Windows
cp src/*.nnue build/ 2>/dev/null || true          # Unix

# 5. Run tests
cd build && ./metalfish_tests                      # Unix
cd build/Release && ./metalfish_tests.exe          # Windows

# 6. Run perft
python3 tests/testing.py --quick

# 7. UCI test
printf "uci\nisready\nposition startpos\ngo depth 5\nquit\n" | ./metalfish
```

## Important Development Facts

### Build System Details
- Uses CMake 3.20+ with separate compilation units
- Link-Time Optimization enabled (`-flto`) - harmless LTO warnings are normal
- NNUE files embedded at link time via assembly directives (must exist before linking)
- Metal shaders compiled via `xcrun metal` on macOS (automatic)
- CUDA separable compilation enabled for device code

### Code Style & Pre-commit
- **C++**: clang-format (configured in `.clang-format` - use it!)
- **Python**: black + isort with --profile=black
- **CMake**: cmake-format
- Run `pre-commit run --all-files` before committing

### Key Dependencies
- **Metal**: Requires macOS 12.0+, metal-cpp (auto-downloaded)
- **CUDA**: CUDA Toolkit 12.0+ (optional, for NVIDIA GPUs)
- **Frameworks (macOS)**: Metal, Foundation, Accelerate, CoreFoundation, QuartzCore
- **Threading**: pthreads (Linux/macOS), Windows threads
- **Python tests**: python-chess library (`pip3 install chess`)

### Search Commands (UCI)
- `go depth N` - Alpha-Beta search to depth N
- `mctsmt movetime M` - Multi-threaded MCTS search for M milliseconds
- `parallel_hybrid movetime M` - Parallel hybrid MCTS+AB search

### Known Quirks
1. **NNUE files are large** (104MB + 3.4MB) and in .gitignore - must download every time
2. **CUDA tests skip in CI** - CUDA builds compile but don't run tests (no GPU in CI)
3. **Metal tests skip on non-macOS** - Expected behavior
4. **Windows needs NNUE files in build/Release/** - CI copies them automatically
5. **LTO warnings** - "lto-wrapper: warning: using serial compilation" is harmless

## Validation Checklist

**Before committing code changes:**
1. Download NNUE files if not present
2. Clean build: `rm -rf build && mkdir build && cd build`
3. Configure with appropriate flags
4. Build: `ninja metalfish` or `cmake --build . -j$(nproc)`
5. Run tests: `./metalfish_tests`
6. Run perft: `python3 tests/testing.py --quick`
7. Test UCI: `printf "uci\nquit\n" | ./metalfish`
8. Run pre-commit hooks: `pre-commit run --all-files` (if pre-commit installed)

**For GPU-related changes (Metal/CUDA):**
- Test on appropriate platform (macOS for Metal, Linux with NVIDIA GPU for CUDA)
- CPU fallback must still work (`-DUSE_METAL=OFF -DUSE_CUDA=OFF`)

## Quick Reference Commands

**Build from scratch:**
```bash
cd src && curl -L -O https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue && curl -L -O https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue && cd .. && mkdir -p build && cd build && cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja metalfish
```

**Run all tests:**
```bash
cd build && ./metalfish_tests && cd .. && python3 tests/testing.py --quick
```

**Check what's built:**
```bash
cd build && ls -lh metalfish* *.nnue *.metallib 2>/dev/null
```

## Trust These Instructions

These instructions were created by thoroughly testing the build process, running all tests, examining the CI pipeline, and validating every command. Only search for additional information if:
- These instructions are incomplete for your specific task
- You encounter an error not documented here
- You need details about a specific algorithm or implementation

For general build, test, and validation tasks, **trust and follow these instructions exactly** - they are accurate and tested.
