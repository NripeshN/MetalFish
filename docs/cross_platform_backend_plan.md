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
| Linux CPU build/test | `cloudbuild/linux-cpu.yaml` | `885e7aa7-19ca-47c0-80f7-842d2c934b0b` |
| CUDA entrypoint compile/test | `cloudbuild/cuda-entrypoint.yaml` | `39a5467f-a249-440a-a4ca-0d698b18fb62` |
| CUDA GPU runtime gate | `tools/run_gcp_cuda_gpu_gate.sh` | manual T4 pass, 2026-05-20, workspace reuse gate |
| GitHub portable Linux/Windows CPU | `.github/workflows/portable-ci.yml` | `26139638867` |

Current CUDA backend boundary:

- `src/nn/cuda/cuda_executor.*` is the inference execution seam.
- `src/nn/network_execution_plan.*` builds the shared ordered and resolved
  tensor/stage plan that CUDA and future portable backends will execute.
  Resolution now infers dense, attention, feed-forward, smolgen, positional,
  and attention-policy matrix shapes from flat BT4 protobuf tensors while
  preserving explicit tensor dimensions when present.
- The CUDA executor seam receives the resolved plan and uploaded weight buffers,
  so real kernels can index device tensors without backend-local name lookups.
- CUDA weight upload preserves empty optional inventory entries as zero-byte
  device tensor views, keeping resolved inventory indices stable across common
  backend contracts and future network variants.
- `src/nn/cuda/cuda_kernels.*` contains tested CUDA compute primitives for
  dense-affine projections/heads, last-axis layer normalization, shared
  elementwise activation functions, input-embedding gate multiply/add, scaled
  residual addition, and fused residual layer normalization for transformer
  attention/FFN blocks. It also contains the first attention-core kernels for
  scaled QK scores, row softmax, and probability-weighted value context
  construction, smolgen attention-bias plus softmax fusion, and
  attention-policy scratch-to-1858 mapping. The CUDA policy map uses cuBLAS for
  the 64x64 square-policy logits, a narrow CUDA kernel for promotion logits, and
  a fixed gather table kept in device constant memory and uploaded once per
  active CUDA device. CUDA feed-forward stages fuse dense1 bias addition with
  activation while preserving the biased dense scratch tensor, and smolgen dense
  stages use the same bias-activation fusion while preserving diagnostic dense
  tensors. CUDA also has the BT4
  dynamic-position-input kernels that expand packed plane masks/values to NHWC
  board rows, gather the first 12 planes for dense positional encoding, and
  concatenate generated positional channels back onto the 112 input channels.
- `src/nn/cuda/cuda_workspace.*` owns reusable per-network execution scratch
  slots for dense, activation, and normalization intermediates. The executor
  seam receives the workspace and its non-blocking stream so future production
  kernels can avoid per-batch `cudaMalloc`/`cudaFree` and device-wide
  synchronization between adjacent inference stages.
- `src/nn/cuda/cuda_buffers.*` exposes stream-aware packed-input upload,
  output clear, and output download paths while preserving synchronous defaults
  for smoke tests and fallback call sites.
- `src/nn/cuda/cuda_input_packing.*` has both contiguous raw packing for
  fixtures and per-position host batch packing for `Network::EvaluateBatch()`,
  so CUDA batching does not depend on unrelated `InputPlanes` objects sharing
  one raw allocation.
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
- `CreateResolvedCudaExecutor()` is now the production CUDA executor shell:
  `CudaNetwork` validates a fully supported resolved schedule, validates named
  output mapping, uploads weights, and installs this executor instead of the
  missing placeholder. The executor uses the same resolved tape, stage input
  derivation, packed input mask/value buffers, stage sequence launcher, and
  named output copier as the smoke path. GPU-device parity against Metal/Lc0 is
  still a release-blocking gate before CUDA can be called strength-ready.
- `CudaNetwork` runs a suppressed-profile batch-1 warmup immediately after
  installing the resolved executor. This primes CUDA kernels, cuBLAS handles,
  and fixed-shape workspaces before the first real search eval. On the 2026-05-20
  manual T4 gate, the first profiled BT4 eval was `16.4 ms` after warmup,
  compared with the previous cold path around `86 ms`; subsequent profiled
  BT4 evals remained in the same warm range.
- The CUDA residual-layernorm path now fuses residual scratch write and
  normalization into one kernel for attention and FFN blocks. The 2026-05-20 T4
  gate kept reference-output parity and moved profiled BT4 evals from roughly
  `16.4/18.25 ms` to `15.9/17.8-18.1 ms`.
- The CUDA FFN dense1 path now fuses bias addition and activation into one
  kernel while leaving the dense pre-activation buffer biased for diagnostics.
  The 2026-05-20 T4 gate kept fixed BT4 outputs, batch parity, and UCI smoke
  green; profiled BT4 evals were `17.85-17.94 ms` with the feed-forward bucket
  around `4.87-4.90 ms`.
- The CUDA smolgen attention path now fuses attention-bias application and
  softmax while preserving the biased score scratch tensor for diagnostics. The
  2026-05-20 T4 gate rejected the first register-max variant for strict
  attention-smoke drift, then accepted the scratch-preserving variant with fixed
  BT4 outputs, batch parity, and UCI smoke green. Steady profiled BT4 evals were
  `17.74-17.76 ms`, with the attention bucket around `10.95 ms`.
- The CUDA smolgen dense path now uses the same bias-activation fusion for both
  smolgen dense layers, preserving the checked dense and activation tensors. The
  2026-05-20 T4 gate kept CUDA smoke, fixed BT4 outputs, batch parity, and UCI
  smoke green; steady profiled BT4 evals were `17.57-17.60 ms`.
- `CudaNetwork` now keeps its execution workspace across batch-size changes and
  lets named CUDA scratch buffers grow on demand. This avoids `cudaFree` /
  `cudaMalloc` churn when MCTS alternates between tactical low-latency batches
  and fuller queued batches. The 2026-05-20 T4 gate kept CUDA smoke, fixed BT4
  outputs, batch parity, and UCI smoke green.
- A CUDA Q/K/V bias fusion attempt was rejected on the 2026-05-20 T4 gate after
  fixed-output drift in the castling-rights reference case, and was reverted.
- The CUDA pipeline smoke now instantiates `CreateResolvedCudaExecutor()` with
  a resolved schedule and named output mapping, so a real NVIDIA-device test
  exercises the same executor class that `CudaNetwork` installs.
- `tools/run_cuda_gpu_gate.sh` is the reusable NVIDIA-host gate. It verifies
  `nvidia-smi` and `nvcc`, builds CUDA with BT4 weights, runs CUDA unit tests,
  runs `test_nn_comparison` through `NNBackend=auto` on the CUDA host, and runs
  a one-thread `NNBackend=cuda` MCTS UCI smoke. Dependency installation waits
  and retries around apt/dpkg locks and refreshes the package index before each
  install attempt so fresh cloud images do not fail the gate while unattended
  upgrades or transient mirror failures are still running.
- `tools/run_gcp_cuda_gpu_gate.sh` creates an ephemeral GCP L4 VM from a clean
  `git archive`, runs `tools/run_cuda_gpu_gate.sh` on the VM, and deletes the
  VM by default. It uses explicit `METALFISH_GCP_*` variables so the current
  local gcloud project cannot accidentally steer the CUDA gate. The default
  zone list covers central, east, west, and northamerica L4 zones to avoid
  treating temporary stockouts as engine failures.
- `CreatePlanSmokeCudaExecutor()` remains available for narrow executor
  diagnostics and can run a tiny resolved-plan pipeline through
  uploaded device weights and real dense/activation/layernorm kernels without
  depending on production weights. Its smoke coverage includes named output
  mapping, explicit branching from a shared body output, chained
  dense/layernorm stages, dense-only stages, input gates, feed-forward residual
  layernorm, attention-policy mapping, non-output positional encoding metadata,
  multi-row batches, and strided policy/value/moves-left/raw-policy writes.
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
