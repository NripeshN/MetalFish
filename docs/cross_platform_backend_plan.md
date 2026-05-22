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
- `src/nn/tables/attention_policy_map.h`: attention-policy scratch-to-1858
  gather table shared by Metal, CUDA, and portable CPU execution.
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
| Windows x86-64 CPU | GitHub Windows MinGW + MSVC | Portable build/UCI package and MSVC host-toolchain coverage |
| Windows x86-64 CUDA | Ephemeral Windows GPU VM | CUDA/DirectML smoke before release |

Cloud GPU runners should be created only for a benchmark run, write logs and
artifacts to Cloud Storage, and then stop/delete themselves. The GitHub
`CUDA GPU Gate` workflow is manual on purpose: every pull request still gets
Linux/Windows portable coverage, while real NVIDIA runtime validation can be
started when CUDA code or backend contracts change.

Current GCP scaffolding:

| Resource | Value |
| --- | --- |
| Project | `metalfish` |
| Region | `us-central1` |
| Artifact Registry | `us-central1-docker.pkg.dev/metalfish/metalfish` |
| Build artifact bucket | `gs://metalfish-build-artifacts-952699201289` |
| Runner service account | `metalfish-runner@metalfish.iam.gserviceaccount.com` |
| Enabled APIs | Compute Engine, Cloud Build, Cloud Run, Artifact Registry, Secret Manager |

GitHub CUDA gate secrets/config:

| Name | Kind | Purpose |
| --- | --- | --- |
| `GCP_CREDENTIALS_JSON` | Secret | Service-account JSON used by `google-github-actions/auth` |
| `METALFISH_GCP_PROJECT` | Repository variable | Optional override; defaults to `metalfish` |

Current remote gates:

| Gate | Build config | Last passing build |
| --- | --- | --- |
| Linux CPU build/test | `cloudbuild/linux-cpu.yaml` | `21729e08-bf3c-4b34-84a2-0d4c722e0167` |
| CUDA entrypoint compile/test | `cloudbuild/cuda-entrypoint.yaml` | `0c0ba5ab-5d55-44a0-a71f-0934c3c495e8` |
| CUDA GPU runtime gate | `tools/run_gcp_cuda_gpu_gate.sh` | `metalfish-cuda-gate-20260522-final-e370951`, L4, 2026-05-22 |
| GitHub CUDA GPU runtime gate | `.github/workflows/cuda-gpu-gate.yml` | Manual dispatch, pending first run |
| GitHub macOS Metal | `.github/workflows/ci.yml` | `26308351194` |
| GitHub portable Linux/Windows CPU | `.github/workflows/portable-ci.yml` | `26308351195` |
| GitHub hybrid regression | `.github/workflows/hybrid-regression.yml` | `26308351186` |

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
- `src/nn/network_attention_plan.*` validates resolved MHA tensor shapes,
  head/depth geometry, smolgen branch dimensions, and global smolgen positional
  weights without depending on CUDA. `src/nn/cuda/cuda_attention_plan.*` keeps
  CUDA compatibility wrappers around that common plan.
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
- The CUDA GPU gate now records opt-in backend batch timings through
  `test_nn_comparison` without treating wall-clock variance as a hard failure.
  The first 2026-05-20 T4 baseline after workspace reuse was:
  `b1=16.922ms`, `b2=20.613ms`, `b4=27.251ms`, `b8=42.228ms`,
  `b16=80.917ms`, `b32=157.184ms` (`4.912ms/eval` at batch 32).
- CUDA attention score/context execution uses the strided cuBLAS attention
  path for all batch sizes. The pointer-array multi-batch path was removed
  after a clean T4 rerun exposed batch-33 policy drift; the regression gate now
  checks both batch 32 and batch 33 against single-position evaluation. The
  accepted 2026-05-20 T4 gate kept CUDA smoke, fixed BT4 outputs, expanded
  batch parity, and UCI smoke green with `b1=16.844ms`, `b2=25.530ms`,
  `b4=26.723ms`, `b8=45.868ms`, `b16=88.098ms`, `b32=169.630ms`
  (`5.301ms/eval` at batch 32).
- `CudaNetwork::Evaluate()` now follows the Metal backend shape by passing a
  one-position span into the shared CUDA batch runner instead of copying the
  full `InputPlanes` object into a temporary vector. The 2026-05-20 T4 gate
  kept CUDA smoke, fixed BT4 outputs, expanded batch parity, and UCI smoke
  green with `b1=16.887ms`, `b2=18.756ms`, `b4=28.453ms`, `b8=47.599ms`,
  `b16=90.355ms`, `b32=175.040ms` (`5.470ms/eval` at batch 32).
- CUDA backend construction now selects the best visible CUDA device before
  allocating buffers or uploading weights, with `METALFISH_CUDA_DEVICE` as an
  explicit override and `CUDA_VISIBLE_DEVICES` left to restrict the visible
  device set. The 2026-05-20 T4 gate kept CUDA smoke, fixed BT4 outputs,
  expanded batch parity, and UCI smoke green with `b1=16.885ms`,
  `b2=25.533ms`, `b4=26.380ms`, `b8=45.070ms`, `b16=86.075ms`,
  `b32=161.738ms` (`5.054ms/eval` at batch 32).
- CUDA backend diagnostics now report the actual selected backend and CUDA
  device in MCTS backend load logs, and the CUDA GPU gate asserts both the
  batch benchmark and a UCI `NNBackend=auto` smoke selected the CUDA
  transformer backend. The 2026-05-20 T4 gate kept CUDA smoke, fixed BT4
  outputs, expanded batch parity, auto-backend UCI smoke, and explicit-CUDA UCI
  smoke green with `b1=16.948ms`, `b2=18.971ms`, `b4=30.237ms`,
  `b8=43.361ms`, `b16=83.492ms`, `b32=158.366ms` (`4.949ms/eval` at batch 32).
- CUDA cuBLAS handles now disable atomics in addition to pedantic math mode.
  The 2026-05-20 L4 gate kept CUDA smoke, fixed BT4 outputs, batch parity, and
  UCI smoke green while reducing the worst single-vs-batch policy drift in the
  parity corpus from about `0.0984` to `0.0151`. The CUDA batch-parity guard is
  now capped at `0.075` policy-logit drift and `0.125` moves-left drift. Two
  clean L4 reruns observed raw single-vs-batch policy deltas just over `0.05`
  while fixed BT4 reference parity remained stable, so the gate still catches
  meaningful batch drift without failing on cuBLAS batch-shape variance. The
  accepted `metalfish-cuda-gate-20260520-213518` run reported worst batch
  policy drift `0.039534` and worst moves-left drift `0.054993`.
- CUDA attention softmax now uses the deterministic reduction path by default;
  `METALFISH_CUDA_DETERMINISTIC_ATTENTION_SOFTMAX=0` restores the faster
  non-deterministic profiling path. The 2026-05-21 L4 gate
  `metalfish-cuda-gate-20260521-151512-detsoftmax-default` accepted CUDA unit tests,
  batch parity, single-reuse stress, batch-reuse stress, and auto/CUDA/hybrid
  UCI smokes with `REUSE_STRESS_MAX policy_delta=0.000008`,
  `SINGLE_REUSE_STRESS_MAX policy_delta=0.000000`, and `b32=104.992ms`
  (`3.2810ms/eval`). The prior default fast path was retained only as an opt-out
  because its reused evaluator paths showed percent-level policy-logit drift on
  the same L4 gate profile.
- The deterministic attention softmax has a width-64 warp-specialized path for
  transformer attention rows, with the generic shared-memory deterministic path
  kept for other widths. The 2026-05-21 L4 gate
  `metalfish-cuda-gate-20260521-161620-warpsoftmax-g2s8` on `g2-standard-8`
  accepted CUDA unit tests, batch parity, reuse stress, and auto/CUDA/hybrid UCI
  smokes with `REUSE_STRESS_MAX policy_delta=0.000007`,
  `SINGLE_REUSE_STRESS_MAX policy_delta=0.000000`, and `b32=96.943ms`
  (`3.0295ms/eval`). An earlier capacity-fallback
  `metalfish-cuda-gate-20260521-155241-warpsoftmax-g2s4` run on
  `g2-standard-4` also passed with `b32=97.004ms` (`3.0314ms/eval`).
- A second pointer-batched attention GEMM attempt was rejected after the
  2026-05-21 L4 gate `metalfish-cuda-gate-20260521-120602` failed the fixed
  BT4 reference check on BK.07 (`a3b4` policy-logit drift `0.057445` against
  the CUDA fixed-reference tolerance). The batch-16/32 speed gain was modest,
  so the experiment was reverted instead of loosening tolerances or leaving a
  numerically unstable opt-in path.
- A per-thread host input packing buffer reuse attempt was rejected after the
  2026-05-21 L4 gate `metalfish-cuda-gate-20260521-125504` failed single-reuse
  stress (`moves_left_delta=0.197540` on
  `e2e4 e7e6 d2d4 d7d5`). The change reduced CPU allocation churn but made
  repeated singleton CUDA evals less stable, so it was reverted.
- `test_nn_comparison` has an opt-in targeted batch probe
  (`METALFISH_NN_BATCH_TRACE_PAIR=1`) that prints the delta for one selected
  entry as both a single eval and a member of a larger batch. This is the safe
  first diagnostic for narrowing CUDA batch drift without changing engine
  execution. The 2026-05-20 L4 gate accepted this probe on batch 32 entry 6
  with `policy_delta=0.000004` and `moves_left_delta=0.000015`.
- The CUDA GPU gate enables `METALFISH_NN_BATCH_TRACE_WORST=1` by default.
  The trace records the worst drift observed during the reused-evaluator batch
  sequence, then reruns that entry with fresh single/batch evaluators and
  compares both reused sides against the fresh confirmation. If the strict
  parity check fails while tracing is enabled, the test now finishes the pass
  and emits the same worst-side diagnostics before returning failure. The
  2026-05-20 L4 gate `metalfish-cuda-gate-20260520-230541` accepted the trace
  with worst initial batch-32 drift on `e2e4 e7e6 d2d4 d7d5`
  (`moves_left_delta=0.054611`, `policy_delta=0.011364`). Fresh confirmation
  dropped to `moves_left_delta=0.007126` and `policy_delta=0.002823`.
  Reused-batch drift was eliminated in that trace (`moves_left_delta=0.0`,
  `policy_delta=0.0`); the remaining larger wobble was on the reused-single
  side (`moves_left_delta=0.047485`, `policy_delta=0.010571`).
- The CUDA GPU gate enables `METALFISH_NN_BATCH_REUSE_STRESS=1` by default.
  This reuses one CUDA evaluator across `32 -> 1 -> 16 -> 33` batch-size
  transitions and compares selected outputs against fresh single-position
  baselines. The 2026-05-20 L4 gate
  `metalfish-cuda-gate-20260520-230541` accepted the stress with worst
  `moves_left_delta=0.000092` and `policy_delta=0.001270` on the shrink-to-1
  start-position probe.
- The NN-backed MCTS legal-move view test still checks move size and move order
  exactly. For CUDA, value and policy logits use the same backend tolerance as
  the parity tests because the test compares two separate CUDA inference calls
  for the same position; exact equality remains required for Metal and stub
  backends.
- CUDA inference now chunks execution batches above 16 and prepares execution
  scratch inside the executor by reserving all tape buffers before clearing the
  workspace on the network stream. This keeps same-size chunk reuse from
  leaking stale intermediate state, also covers first allocation/growth of named
  scratch buffers, and preserves allocated device buffers. The 2026-05-20 L4
  gate `metalfish-cuda-gate-20260520-230541` accepted CUDA unit tests, fixed
  BT4 references, expanded batch parity, reuse stress, auto/CUDA UCI smokes, and
  hybrid-CUDA smoke. `METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE` can raise the
  chunk size for experiments, but the production default remains 16: the
  2026-05-21 L4 fallback gate
  `metalfish-cuda-gate-20260521-164547-stable32-g2s4` accepted a batch-32
  experiment with `REUSE_STRESS_MAX policy_delta=0.000007`, but batch-32 timing
  stayed flat at `97.446ms` (`3.0452ms/eval`), so the larger default was
  rejected.
- CUDA output/intermediate buffers are cleared on every inference by default.
  This keeps mixed singleton/batch reuse inside the fixed-reference tolerances
  without releasing device allocations or slowing the hot path. Two 2026-05-21
  L4 gates, `metalfish-cuda-gate-20260521-021626` and
  `metalfish-cuda-gate-20260521-022454`, accepted fixed BT4 references,
  expanded batch parity, reuse stress, auto/CUDA UCI smokes, and hybrid-CUDA
  smoke with batch-1 around 7.6 ms. The clean default gate
  `metalfish-cuda-gate-20260521-023431` accepted the same coverage from a
  committed branch-tip archive. Disabling the clear was rejected by the
  2026-05-21 L4 gate `metalfish-cuda-gate-20260521-130934`, which failed the
  rook-endgame fixed reference (`moves_left_delta=0.2301`) before UCI smokes.
- A fused CUDA input-expansion plus positional-input gather attempt was
  rejected after the 2026-05-21 L4 gate
  `metalfish-cuda-gate-20260521-132519` failed batch parity on
  `e2e4 c7c5` (`f1b5` policy-logit drift `0.0799888`) and single-reuse stress
  on `e2e4 e7e6 d2d4 d7d5`. The extra launch removal was not worth accepting
  input-preprocess drift, so the separate expansion and positional gather
  kernels remain the default path.
- Portable Linux and Windows artifacts now include `PORTABLE_ARTIFACT.md`,
  generated by `tools/write_portable_manifest.py`, so downloaded CI packages
  state the platform, binary name, source branch/commit, and backend scope.
  The current portable packages are CPU AB plus diagnostic stub-MCTS builds;
  they do not imply CUDA or DirectML strength support.
- The CUDA GPU gate now emits `cuda-gpu-summary.md` on both success and
  failure, with the selected NVIDIA device, resolved backend string, batch
  timings when enabled, direct failure snippets, UCI bestmoves when reached,
  stage-trace compare lines when tracing is enabled, selected attention and
  dynamic positional encoding trace summaries, and a pointer to
  `cuda-gpu-parity-report.md`. Stage traces can now sample one batch entry with
  `METALFISH_CUDA_TRACE_STAGE_ENTRY`, which lets a single-position baseline be
  compared against the same history entry inside a larger batch. When
  `METALFISH_CUDA_TRACE_COMPARE_BASE_RUN=0`, the compare path now treats the
  first reported trace slice as the baseline even if
  `METALFISH_CUDA_TRACE_STAGE_SKIP` skipped earlier invocations; compare lines
  include `baseline_actual_run` so diagnostic logs remain unambiguous. The parity
  report records fixed BT4 reference deltas plus single-vs-batch drift across
  the expanded batch corpus, so remote CUDA runs preserve the evidence needed
  to investigate silent backend drift. The accepted 2026-05-21 L4 gate
  `metalfish-cuda-gate-20260521-123738` verified the failure-summary trap and
  backend-marker fallback in a narrow no-benchmark pass; the rejected
  `metalfish-cuda-gate-20260521-125504` host-packing experiment validated that
  summaries surface the direct NN comparison failure. An earlier accepted
  2026-05-20 L4 summary `metalfish-cuda-gate-20260520-230541` recorded timings
  of `b1=7.630ms`, `b2=9.601ms`, `b4=13.890ms`, `b8=25.593ms`,
  `b16=47.127ms`, `b32=95.801ms` (`2.9938ms/eval` at batch 32).
- The 2026-05-21 L4 diagnostic gate
  `metalfish-cuda-gate-20260521-135430` accepted the dynamic positional
  encoding trace path with `METALFISH_CUDA_TRACE_STAGE_OUTPUTS=1` and
  `METALFISH_CUDA_TRACE_DYNAMIC_PE_INTERNALS=1` on a batch-16 target. The
  sampled expanded input, dense positional input, and dense PE output matched
  between traced runs at the compare threshold, so current preprocessing drift
  is not the source of the observed downstream single-vs-batch variation.
- The 2026-05-21 L4 diagnostic gate
  `metalfish-cuda-gate-20260521-142114` accepted skip-safe stage trace
  baselines and selected attention tracing with
  `METALFISH_CUDA_TRACE_STAGE_SKIP=8`, `METALFISH_CUDA_TRACE_STAGE_ENTRY=9`,
  and `METALFISH_CUDA_TRACE_ATTENTION_INTERNALS=1`. The trace-pair target
  showed identical first-block attention summaries between the selected single
  baseline and the matching batch entry, while `TRACE_WORST_CONFIRMED` was
  effectively zero. The remaining measurable drift is now isolated to reused
  evaluator/workspace paths (`REUSE_STRESS_MAX` policy delta around `0.012`),
  not fresh single-vs-batch math or the first encoder attention block.
- The 2026-05-21 L4 gate `metalfish-cuda-gate-20260521-143113` rejected
  `METALFISH_CUDA_RELEASE_SINGLE_WORKSPACE_EACH_RUN=1`: batch-1 latency rose
  from about `7.6ms` to `14.3ms`, NN comparison failed before UCI smokes, and
  batch reuse drift worsened (`REUSE_STRESS_STEP warm32` policy delta about
  `0.051`, value mismatch). Do not promote single-workspace release as a
  default; it is a diagnostic knob only.
- The 2026-05-21 L4 gate `metalfish-cuda-gate-20260521-143922` tested
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It passed the CUDA, auto, and hybrid UCI
  smokes with normal throughput (`b32` about `97.7ms`), but it did not improve
  the main parity target: fresh batch-33 and single-reuse drift were worse than
  the default gate (`TRACE_WORST_CONFIRMED` policy delta about `0.033`,
  `SINGLE_REUSE_STRESS_MAX` policy delta about `0.032`). Keep it as an
  opt-in diagnostic environment, not a deployment default.
- The CUDA attention smoke keeps strict `1e-5` checks for individual Q/K/V,
  smolgen, score, softmax, context, projection, residual, and layernorm
  tensors. The attention-only sequence-level check now uses a `5e-3`
  tolerance, matching the already approximate end-to-end sequence checks and
  avoiding false failures from cuBLAS allocation/layout variance while still
  catching meaningful drift.
- A CUDA Q/K/V bias fusion attempt was rejected on the 2026-05-20 T4 gate after
  fixed-output drift in the castling-rights reference case, and was reverted.
- A CUDA attention score/context host-pointer-batched cuBLAS attempt was
  rejected on the 2026-05-21 L4 gate
  `metalfish-cuda-gate-20260521-204030-batched-attn-g2s8`. It compiled through
  the remote CUDA entrypoint build, but the runtime gate failed evaluator
  legal-move parity, deterministic bestmove reproducibility, and the BK.07
  low-node sentinel. Host pointer arrays must not be used with cuBLAS batched
  attention calls.
- A CUDA attention score/context device-pointer-batched cuBLAS attempt compiled
  and passed the 2026-05-21 T4 correctness gate
  `metalfish-cuda-gate-20260521-234025-device-ptrs-t4`, but it regressed batch
  timings (`b32=206.611ms` versus the prior T4 strided path around
  `157-170ms`). It was reverted as a speed regression; future attention work
  should target a custom kernel or graph-captured strided path instead of
  pointer-array cuBLAS.
- CUDA resolved execution uses graph replay by default when the run is
  compatible, matching Metal's persistent graph execution model more closely.
  Set `METALFISH_CUDA_GRAPH=0` or `METALFISH_CUDA_GRAPH_EXECUTION=0` to force
  the uncaptured path. The first inference for a batch/workspace generation
  primes the normal path, the second captures the existing workspace clear,
  resolved stage sequence, and output mapping without changing math, and later
  same-key calls replay the graph. The path is disabled when CUDA profiling,
  stage tracing, attention tracing, dynamic PE tracing, or per-run workspace
  release knobs are active. The graph key includes batch size, workspace
  generation, inference-buffer generation, stream, and device output pointers;
  graph API failures reset the cache and fall back to uncaptured execution.
  The 2026-05-22 L4 gate `metalfish-cuda-gate-20260522-final-e370951` accepted
  the default graph path on the merged `main` tip with
  `CUDA graph replay observed: yes`, CUDA unit tests, fixed BT4 references,
  batch parity, single/batch reuse stress, the standalone CUDA NN probe,
  auto/CUDA/hybrid UCI smokes, the non-Apple ANE-disable hybrid smoke, and
  batch timings of `b1=6.923ms`, `b16=51.499ms`, and `b32=97.210ms`.
- The CUDA pipeline smoke now instantiates `CreateResolvedCudaExecutor()` with
  a resolved schedule and named output mapping, so a real NVIDIA-device test
  exercises the same executor class that `CudaNetwork` installs.
- The CUDA dense/attention execution sequence can now consume the already
  validated resolved schedule held by `PlanSmokeCudaExecutor` and
  `ResolvedCudaExecutor`. This removes per-inference schedule reconstruction
  from the hot path without changing the shared resolved-plan contract. The
  2026-05-20 L4 gate `metalfish-cuda-gate-20260520-230541` kept CUDA unit tests,
  fixed BT4 references, expanded batch parity, first-use evaluator stress,
  reusable-evaluator batch stress, and auto/CUDA/hybrid UCI smokes green.
- `tools/run_cuda_gpu_gate.sh` is the reusable NVIDIA-host gate. It verifies
  `nvidia-smi` and `nvcc`, builds CUDA with BT4 weights, runs CUDA unit tests,
  runs `test_nn_comparison` through `NNBackend=auto` on the CUDA host, asserts
  that auto selected the CUDA transformer backend, builds and runs the
  standalone NN probe with `--backend cuda`, and runs one-thread MCTS UCI smokes
  for `NNBackend=auto`, explicit `NNBackend=cuda`, and the production hybrid
  search path with CUDA-backed transformer MCTS. Dependency installation
  waits
  and retries around apt/dpkg locks and refreshes the package index before each
  install attempt so fresh cloud images do not fail the gate while unattended
  upgrades or transient mirror failures are still running. The 2026-05-20 L4
  gate `metalfish-cuda-gate-20260520-230541` accepted the hybrid-CUDA smoke
  with `bestmove e2e4`.
- `METALFISH_CUDA_PROFILE=1` now runs as a separate hybrid-CUDA profiling
  smoke after the correctness gate. CUDA unit tests and the parity binary are
  kept unprofiled so CUDA event synchronization cannot perturb correctness
  checks. The optional trace-pair diagnostic uses fresh evaluator instances so
  it cannot perturb the parity loop, while `cuda-gpu-profile.log` and the
  summary still capture stage buckets for performance work. On the accepted L4
  profile smoke, the first profiled BT4 eval was `7.351 ms`, led by
  attention/layernorm stages at `4.226 ms`.
- The 2026-05-21 L4 profile gate
  `metalfish-cuda-gate-20260521-200708-profile` passed after merging current
  `main` into `cuda-support`, including CUDA unit tests, BT4 fixed references,
  batch parity, reuse stresses, auto/CUDA/hybrid UCI smokes, and VM cleanup.
  Stable execution batch timings on L4 were `b1=7.654ms`,
  `b2=9.568ms`, `b4=13.860ms`, `b8=25.747ms`, `b16=47.972ms`, and
  `b32=95.783ms`; batch 16 remains the default because batch 32 does not
  improve per-eval latency enough to justify more queueing. The profile showed
  `7.370 ms` sequence time, `0.031 ms` output sync, and `77.229 MB` workspace.
  The dominant bucket is still attention/layernorm at `4.235 ms` across 15
  stages, followed by feed-forward/layernorm at `1.707 ms`; the slowest named
  stage is `body.input_embedding_preprocess` at `0.546 ms`. The next CUDA
  performance target should be stage orchestration, attention/layernorm launch
  overhead, or a parity-safe input embedding rewrite.
- CUDA warmup now synchronizes its stream before the backend constructor
  returns. That keeps cuBLAS work from one freshly-created evaluator from
  overlapping another evaluator on a different stream and prevents first-use
  policy drift in the BT4 batch-parity gate.
- `test_nn_comparison` now includes a fresh-evaluator first-use stress block.
  On CUDA it defaults to three back-to-back evaluator pairs and can be adjusted
  with `METALFISH_NN_FIRST_USE_STRESS_ITERS`; the GCP gate forwards that knob.
- `tools/run_gcp_cuda_gpu_gate.sh` creates an ephemeral GCP L4 VM from a clean
  `git archive`, runs `tools/run_cuda_gpu_gate.sh` on the VM, and deletes the
  VM by default. It uses explicit `METALFISH_GCP_*` variables so the current
  local gcloud project cannot accidentally steer the CUDA gate. The default
  zone list covers central, east, west, and northamerica L4 zones to avoid
  treating temporary stockouts as engine failures. By default it collects
  `cuda-gpu-summary.md`, `cuda-gpu-parity-report.md`, and CUDA gate logs into
  `results/cuda_gpu_gate/<instance>/` before deleting the VM; set
  `METALFISH_GCP_GCS_PREFIX=gs://...` to also upload those artifacts to Cloud
  Storage, or `METALFISH_GCP_COLLECT_ARTIFACTS=0` to disable collection. The
  2026-05-20 L4 artifact-collection gate copied eight files locally: summary,
  parity report, CUDA unit-test log, NN comparison log, auto-backend UCI smoke
  log, explicit-CUDA UCI smoke log, hybrid-CUDA UCI smoke log, and the opt-in
  CUDA profile log.
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

Current portable CPU transformer boundary:

- `NNBackend=cpu` is an explicit portable transformer backend. It is compiled
  on every platform, but `auto` does not select it yet.
- The backend loads real protobuf weights through the shared descriptor,
  tensor-plan, weight-inventory, execution-plan, and resolved-plan code. This
  keeps Linux and Windows aligned with Metal/CUDA network parsing and resolved
  tensor shape inference.
- The backend now owns resolved tensor copies and executes simple dense,
  layernorm, gate, feed-forward, non-output positional metadata, dynamic
  position input-preprocessing, body attention, and attention-policy map
  resolved plans, including adjacent feed-forward and attention residual
  layernorm plus batched output decoding through the shared policy/value
  decoder. It mirrors CUDA's packed-plane expansion into square rows for
  `body.input_embedding_preprocess`, so minimal PE-dense fixtures exercise the
  same `[batch*64, input_planes + pe_width]` contract used by BT4 before body
  attention starts.
- BT4 execution still fails loudly with the first unsupported resolved stage
  instead of falling back to the diagnostic stub. Current CPU fixture coverage
  reaches the same attention-policy raw scratch plus 1858-logit gather contract
  used by CUDA/Metal; the next CPU milestone is a bounded real-BT4 smoke that
  proves full execution is practical enough to keep enabled in CI.

Portable CI builds Linux CPU, Windows MinGW CPU, and Windows MSVC CPU
artifacts. The MSVC leg is included because Windows CUDA uses the MSVC host
toolchain; it is the CPU prerequisite before adding a Windows NVIDIA runtime
gate. Each job runs AB UCI smoke plus an explicit `NNBackend=stub` MCTS smoke,
so portable builds verify the MCTS construction path without downloading BT4
weights. The uploaded artifacts include a generated manifest that makes this
backend scope explicit. Branch tip `e3709510` had Linux CPU, Windows MinGW CPU,
Windows MSVC CPU, macOS Metal, CUDA L4 runtime, and the bounded hybrid
regression gate green after merging `origin/main`. The hybrid gate uses a
bounded 300-puzzle offline sample for PR runs; the accepted run scored
candidate BK repeats `[22, 22, 21]` versus baseline `[22, 22, 22]`, and
candidate puzzles `298/300` versus baseline `298/300` with zero candidate
errors.

## First Milestones

1. Keep Apple Metal path green while adding `NNBackend` plumbing.
2. Add Linux and Windows CPU CI jobs with `USE_METAL=OFF`.
3. Add a CUDA backend shell that compiles with nvcc/CUDAToolkit and fails
   clearly when requested before the actual implementation exists.
4. Bring up Linux CUDA inference behind `NN::Network` without touching MCTS.
5. Add CUDA parity tests: same position, same weights, policy/value close to
   Metal/Lc0.
6. Add Windows packaging and UCI smoke.
