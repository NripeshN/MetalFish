# Cross-Platform Backend Plan

MetalFish should keep one search engine and one neural-network contract across
all platforms. Platform-specific code should live behind backend interfaces so
Apple Silicon, CUDA/Linux, and Windows can improve independently without
forking MCTS, hybrid arbitration, UCI, or benchmark tooling.

## Regression Rule

Every backend change must start from a recorded baseline and rerun the smallest
benchmark that covers the touched surface.

| Surface | Baseline before change | Required rerun after change |
| --- | --- | --- |
| CMake/UCI/common code | `cmake --build build --target metalfish -j2`, `python3 tests/test_benchmark_configs.py` | Same commands, plus UCI smoke |
| MCTS backend plumbing | BK.07 `go nodes 50` with BT4 weights | Same smoke; must return `h5f6` |
| Linux CPU build | Linux release configure/build/tests | Same on Linux runner |
| CUDA backend | Linux CPU build plus CUDA backend smoke | Same plus MCTS/Lc0 parity subset |
| Windows build | Windows release configure/build/UCI smoke | Same on Windows runner |
| Strength tuning | Current tactical or tournament baseline | Same benchmark with identical resources |

If a change regresses the relevant baseline, revert it before continuing.

## Backend Contract

Common code owns:

- `src/nn/encoder.*`: Lc0-compatible input planes.
- `src/nn/input_plane_packing.h`: shared packed sparse input contract used by
  Metal and CUDA.
- `tests/nn_input_fixture.*`: backend-neutral packed input fixture with
  explicit plane masks/values for regression tests.
- `src/nn/policy_map.*`: 1858-policy move mapping.
- `src/nn/loader.*` and `src/nn/weights.*`: protobuf weight loading.
- `src/nn/network.h`: platform-neutral `NN::Network` inference interface.
- `src/nn/network_weight_inventory.*`: selected policy/value/moves-left tensor
  inventory shared by platform backends before device upload.
- `src/nn/network_output_decoder.*`: shared policy/value/moves-left output
  decoding used by Metal and future CUDA/Windows backends.
- `src/mcts/*`: MCTS search, cache, stoppers, and backend adapter.
- `src/hybrid/*`: CPU/GPU coordination and final move arbitration.

Platform code should own only the implementation of `NN::Network`:

| Backend | Platform | Status | Intended role |
| --- | --- | --- | --- |
| `metal` | macOS/Apple Silicon | Production | MPSGraph BT4 inference |
| `cuda` | Linux/Windows NVIDIA | Toolchain-gated entrypoint | CUDA/TensorRT or ONNX Runtime CUDA BT4 inference |
| `directml` | Windows GPUs | Planned | Windows fallback where CUDA is unavailable |
| `cpu` | Any | Planned | Correctness fallback, not strength target |
| `stub` | Any | Existing diagnostic fallback | Tests only; never a strength backend |

The UCI option `NNBackend` is the common selector. `auto` should choose the
strongest available backend for the host. Explicit backend names are for
benchmarking and diagnostics.

## Cloud Test Matrix

The goal is to match common chess-engine tournament environments while keeping
cloud spend controlled.

| Matrix leg | Suggested runner | Purpose |
| --- | --- | --- |
| macOS arm64 Metal | GitHub-hosted macOS | Release path and Apple parity |
| Linux x86-64 CPU | GitHub Ubuntu or GCP C3/N2 | AB build/test and portable UCI |
| Linux x86-64 CUDA | GCP `g2-standard-8` + NVIDIA L4 | CUDA backend correctness and NPS |
| Windows x86-64 CPU | GitHub Windows | Portable build/UCI package |
| Windows x86-64 CUDA | Ephemeral Windows GPU VM | CUDA/DirectML smoke before release |

Cloud GPU runners should be created only for a benchmark run, write logs and
artifacts to Cloud Storage, and then stop/delete themselves.

Current GCP scaffolding:

| Resource | Value |
| --- | --- |
| Project | `metalfish` |
| Region | `us-central1` |
| Artifact Registry | `us-central1-docker.pkg.dev/metalfish/metalfish` |
| Build artifact bucket | `gs://metalfish-build-artifacts-952699201289` |
| Runner service account | `metalfish-runner@metalfish.iam.gserviceaccount.com` |
| Enabled APIs | Compute Engine, Cloud Build, Cloud Run, Artifact Registry, Secret Manager |

Current remote gates:

| Gate | Build config | Last passing build |
| --- | --- | --- |
| Linux CPU build/test | `cloudbuild/linux-cpu.yaml` | `299e6c64-ea84-4082-b387-b2a84c1b5948` |
| CUDA entrypoint compile/test | `cloudbuild/cuda-entrypoint.yaml` | `337a5901-6b58-4c42-b56b-739a54cdbd28` |

Current CUDA backend boundary:

- `src/nn/cuda/cuda_executor.*` is the inference execution seam.
- Production CUDA still installs a missing executor and refuses real inference.
- `CreateNullCudaExecutorForSmoke()` exercises packed inputs, device buffers,
  output downloads, and shared output decoding without pretending strength
  inference is implemented.

## First Milestones

1. Keep Apple Metal path green while adding `NNBackend` plumbing.
2. Add Linux and Windows CPU CI jobs with `USE_METAL=OFF`.
3. Add a CUDA backend shell that compiles with nvcc/CUDAToolkit and fails
   clearly when requested before the actual implementation exists.
4. Bring up Linux CUDA inference behind `NN::Network` without touching MCTS.
5. Add CUDA parity tests: same position, same weights, policy/value close to
   Metal/Lc0.
6. Add Windows packaging and UCI smoke.
