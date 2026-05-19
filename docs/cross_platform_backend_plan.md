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
| Linux CPU build/test | `cloudbuild/linux-cpu.yaml` | `6b0aaace-fa62-4e41-bb7e-06dd906f0628` |
| CUDA entrypoint compile/test | `cloudbuild/cuda-entrypoint.yaml` | `a7d4684a-ca31-4cd8-9165-ee682feda787` |
| GitHub portable Linux/Windows CPU | `.github/workflows/portable-ci.yml` | `26070306694` |

Current CUDA backend boundary:

- `src/nn/cuda/cuda_executor.*` is the inference execution seam.
- `src/nn/network_execution_plan.*` builds the shared ordered and resolved
  tensor/stage plan that CUDA and future portable backends will execute.
- The CUDA executor seam receives the resolved plan and uploaded weight buffers,
  so real kernels can index device tensors without backend-local name lookups.
- `src/nn/cuda/cuda_kernels.*` contains tested CUDA compute primitives for
  dense-affine projections/heads, last-axis layer normalization, shared
  elementwise activation functions, input-embedding gate multiply/add, and
  scaled residual addition for feed-forward normalization. It also contains
  the first attention-core kernels for scaled QK scores, row softmax, and
  probability-weighted value context construction, smolgen attention-bias
  addition before attention softmax, and attention-policy scratch-to-1858
  mapping.
- `src/nn/cuda/cuda_workspace.*` owns reusable per-network execution scratch
  slots for dense, activation, and normalization intermediates. The executor
  seam receives the workspace and its non-blocking stream so future production
  kernels can avoid per-batch `cudaMalloc`/`cudaFree` and device-wide
  synchronization between adjacent inference stages.
- `src/nn/cuda/cuda_buffers.*` exposes stream-aware packed-input upload,
  output clear, and output download paths while preserving synchronous defaults
  for smoke tests and fallback call sites.
- `src/nn/cuda/cuda_execution_tape.*` binds resolved execution steps to named
  intermediate device buffers. The current smoke executor uses it for
  dense/activation/normalization intermediates, including multi-stage
  dense/layernorm, gate, and feed-forward/layernorm sequences; production CUDA
  layers should extend this rather than allocating anonymous scratch. Attention
  tape bindings now reserve explicit Q/K/V, score, probability, context, output
  projection, residual, and smolgen compress/dense/norm/global-bias scratch
  using resolved head and square geometry. Attention-policy heads reserve both
  raw scratch logits and mapped 1858-policy logits.
- `src/nn/cuda/cuda_execution_schedule.*` classifies resolved plan steps into
  supported dense/activation stages, supported dense/layernorm stages,
  supported gate stages, supported adjacent attention/layernorm stages,
  supported attention/smolgen/layernorm stages,
  supported feed-forward stages, supported non-output positional encoding
  metadata stages, supported attention-policy map stages, CUDA-managed
  boundaries, and explicit unsupported operations before any kernels launch.
- `src/nn/cuda/cuda_plan_analysis.*` provides the shared CUDA-local view of
  resolved stage groups, dense stage widths, value-error exclusion, and last
  body/head stage discovery. Stage execution and output mapping use this helper
  so head branching and output source selection stay aligned.
- `src/nn/cuda/cuda_attention_plan.*` validates resolved MHA tensor shapes,
  head/depth geometry, smolgen branch dimensions, and global smolgen positional
  weights before attention kernels are allowed into the executor path.
- `src/nn/cuda/cuda_stage_executor.*` owns reusable dense/activation/layernorm
  stage execution, input-embedding gate execution, feed-forward residual
  layernorm execution, attention Q/K/V projection launches, attention
  score/softmax/context execution, attention output projection launches,
  attention residual layernorm execution, smolgen compress/dense/layernorm
  execution with global positional bias injection, attention-policy map
  execution, and strided device-row
  copies, so smoke and production CUDA executors share the same launch path. It
  derives CUDA-local stage input bindings from the
  resolved plan and schedule, allowing independent heads to branch from the
  last supported body output instead of forcing every stage into one linear
  chain.
- `src/nn/cuda/cuda_output_mapping.*` maps named executed CUDA stages to
  policy, value, moves-left, and raw-policy output buffers. This keeps output
  ownership explicit instead of treating the last executed stage as every
  output tensor. Output sources are selected from compatible scheduled head
  stages by target width, so the smoke and production paths do not depend on
  one fixed CUDA stage name per head. Attention-policy maps are preferred as
  policy sources and suppress redundant raw-policy output binding because the
  CUDA stage already performs the final scatter.
- `CreatePlanSmokeCudaExecutor()` runs a tiny resolved-plan pipeline through
  uploaded device weights and real dense/activation/layernorm kernels without
  enabling production CUDA inference prematurely. Its smoke coverage includes
  named output mapping, explicit branching from a shared body output, chained
  dense/layernorm stages, dense-only stages, input gates, feed-forward residual
  layernorm, attention-policy mapping, non-output positional encoding metadata,
  multi-row batches, and strided policy/value/moves-left/raw-policy writes.
- Production CUDA still installs a missing executor and refuses real inference.
- `CreateNullCudaExecutorForSmoke()` exercises packed inputs, device buffers,
  output downloads, and shared output decoding without pretending strength
  inference is implemented.

Portable CI now builds Linux CPU and Windows MinGW CPU artifacts. Both jobs run
AB UCI smoke plus an explicit `NNBackend=stub` MCTS smoke, so portable builds
verify the MCTS construction path without downloading BT4 weights.

## First Milestones

1. Keep Apple Metal path green while adding `NNBackend` plumbing.
2. Add Linux and Windows CPU CI jobs with `USE_METAL=OFF`.
3. Add a CUDA backend shell that compiles with nvcc/CUDAToolkit and fails
   clearly when requested before the actual implementation exists.
4. Bring up Linux CUDA inference behind `NN::Network` without touching MCTS.
5. Add CUDA parity tests: same position, same weights, policy/value close to
   Metal/Lc0.
6. Add Windows packaging and UCI smoke.
