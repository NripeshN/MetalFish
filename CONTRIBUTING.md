# Contributing to MetalFish

Thanks for your interest in MetalFish. The project is licensed under GPL-3.0 and
contributions are accepted under the same license.

## Prerequisites

Download the network files before building (CMake copies the NNUE files into
`build/` at configure time, so they must exist first):

```bash
python3 tools/download_engine_networks.py --dest networks
```

## Build

macOS (Apple Silicon — the primary path):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
cmake --build build --target metalfish metalfish_tests -j"$(sysctl -n hw.ncpu)"
```

For Linux/Windows CUDA, see the README and `docs/cross_platform_backend_plan.md`.

## Test

```bash
./build/metalfish_tests            # C++ unit suite: core search eval_gpu mcts hybrid
python3 tests/testing.py --quick   # UCI protocol + perft smoke
python3 tests/test_lichess_bot.py
python3 tests/test_lichess_puzzle_runner.py
```

## Lint / format

`pre-commit` enforces clang-format (C++), black + isort (`--profile=black`,
Python), cmake-format, and check-yaml. Run it before committing:

```bash
pre-commit run --all-files
```

## Pull requests

- Keep changes focused — one logical change per PR.
- Run the unit suite and `pre-commit` locally; CI must be green before merge.
- For search/eval behavior changes, include validation evidence. The engine is
  already strong, so gains are incremental and need low-noise measurement
  (cutechess tournaments and/or the Lichess bot), not single positions.
- Strength claims must follow the README "Benchmarking Policy" and "Strength
  Claims" sections — no "stronger than X" wording without large-scale testing.

## Acknowledgements

MetalFish derives from Stockfish (alpha-beta search and NNUE evaluation) and is
Lc0-compatible (MCTS and the BT4 transformer), both GPL-3.0. See the README
"Acknowledgements" section.
