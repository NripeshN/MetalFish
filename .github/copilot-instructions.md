# MetalFish Copilot Instructions

MetalFish v0.1.0-alpha is a hybrid UCI chess engine. Treat the Hybrid engine
as the primary product surface.

## Platform Surface

- macOS on Apple Silicon ARM64 with Metal/MPSGraph is the primary release path.
- Linux x86-64 NVIDIA CUDA and Windows x86-64 MSVC/CUDA are parity-gated
  platform paths.
- Portable Linux/Windows CPU builds are correctness and fallback targets, not
  transformer strength targets.
- C++20, Objective-C++ for Metal/MPSGraph, and CUDA C++ for NVIDIA backends
- CMake 3.20+
- Platform dependencies: `protobuf zlib abseil`, plus CUDA/cuBLAS for CUDA
  builds
- Python tooling dependencies: `python3 -m pip install -r tests/requirements.txt`

## Build

Download networks before building:

```bash
mkdir -p networks
curl -L -o networks/nn-c288c895ea92.nnue \
  https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -L -o networks/nn-37f18f62d772.nnue \
  https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
curl -L -o networks/BT4-1024x15x32h-swa-6147500.pb.gz \
  https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz
gzip -dc networks/BT4-1024x15x32h-swa-6147500.pb.gz \
  > networks/BT4-1024x15x32h-swa-6147500.pb
```

Build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
cmake --build build --target metalfish metalfish_tests -j"$(sysctl -n hw.ncpu)"
```

For CUDA backend work, use the CI/GCP gate scripts and `-DUSE_CUDA=ON` builds.
Do not treat the portable CPU transformer as a CUDA strength substitute.

## Required Checks

Run these before release-oriented changes:

```bash
./build/metalfish_tests
python3 tests/testing.py --quick
python3 tests/test_lichess_bot.py
python3 tests/test_lichess_puzzle_runner.py
python3 tests/test_ponder_stress.py --smoke
```

For tactical search changes, also run:

```bash
python3 tests/paper_benchmarks.py --tactical --movetime 5000 --threads 1
```

## Engine Defaults

The binary remains UCI-compatible with AB available as a fallback. Public
strength testing should configure:

```text
setoption name UseHybridSearch value true
setoption name NNWeights value networks/BT4-1024x15x32h-swa-6147500.pb
```

Do not touch platform-specific backend directories unless the change is
specifically about that backend. Shared NN/search contracts should stay in the
platform-neutral `src/nn/`, `src/mcts/`, and `src/hybrid/` seams.
