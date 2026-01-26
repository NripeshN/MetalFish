# Copilot instructions for this repository

## Repository overview
- MetalFish is a UCI chess engine with Metal-accelerated NNUE evaluation and three search modes: Alpha-Beta, MCTS, and a parallel hybrid.
- Key directories:
  - `src/core`: board representation, move generation, and types
  - `src/search`: alpha-beta search logic
  - `src/mcts`: MCTS and hybrid integration
  - `src/gpu`: Metal GPU backend and NNUE GPU integration
  - `tests`: C++ test suite
  - `tools`: tournament and analysis scripts

## Build and test
- Target platform: macOS 12+ with Apple Silicon and Metal toolchain (Xcode 14+ with Metal SDK).
- Standard build:
  ```bash
  mkdir -p build && cd build
  cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
  ninja metalfish
  ```
- Tests:
  ```bash
  cd build
  ninja metalfish_tests
  ./metalfish_tests
  ```
- NNUE files (for engine runs) can be fetched from the README URLs; do not commit these artifacts.

## Style and formatting
- C++: clang-format (pre-commit hook configured).
- Python: black and isort (pre-commit hooks configured).
- CMake: cmake-format (pre-commit hook configured).
- Run `pre-commit run --all-files` where feasible.

## Development notes
- Keep changes minimal and focused on the issue scope.
- Avoid committing generated binaries, network weights, or benchmark outputs.
- GPU/Metal benchmarks are expensive; run only when necessary and on supported hardware.
- No secrets or tokens should be added to the codebase or configuration.
