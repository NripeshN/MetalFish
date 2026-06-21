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

- `src/nn/input_planes.h`: backend-neutral input and policy-output dimensions.
- `src/nn/encoder.*`: Lc0-compatible position-to-plane encoding.
- `src/nn/input_plane_packing.h`: shared packed sparse input contract used by
  Metal and CUDA.
- `tests/nn_input_fixture.*`: backend-neutral packed input fixture with
  explicit plane masks/values for regression tests.
- `src/nn/policy_map.*`: 1858-policy move mapping.
- `src/nn/tables/attention_policy_map.h`: attention-policy scratch-to-1858
  gather table shared by Metal, CUDA, and portable CPU execution.
- `src/nn/weights_file.h`, `src/nn/loader.*`, and `src/nn/weights.*`:
  protobuf weight loading behind a lightweight forward-declared handle.
- `src/nn/network.h`: platform-neutral `NN::Network` inference interface.
- `src/nn/network_weight_inventory.*`: selected policy/value/moves-left tensor
  inventory shared by platform backends before device upload.
- `src/nn/network_output.h`: backend-neutral policy/value/moves-left result
  record.
- `src/nn/network_output_decoder.*`: shared policy/value/moves-left output
  decoding used by Metal, CUDA, and portable backends.
- `src/nn/network_tensor_plan.*`: tensor layout, shape metadata, and decoded
  output target ordering shared by Metal and exposed to CUDA/Windows backends.
- `src/mcts/*`: MCTS search, cache, stoppers, and backend adapter.
- `src/hybrid/*`: CPU/GPU coordination and final move arbitration.

Platform code should own only the implementation of `NN::Network`:

| Backend | Platform | Status | Intended role |
| --- | --- | --- | --- |
| `metal` | macOS/Apple Silicon | Production | MPSGraph BT4 inference |
| `cuda` | Linux/Windows NVIDIA | Linux and Windows parity-gated | CUDA BT4 and legacy inference with shared NN contract |
| `directml` | Windows GPUs | Planned | Windows fallback where CUDA is unavailable |
| `cpu` | Any | Portable fallback | Correctness fallback, not strength target |
| `stub` | Any | Existing diagnostic fallback | Tests only; never a strength backend |

The UCI option `NNBackend` is the common selector. `auto` chooses the strongest
available backend for the host, then falls back to the portable CPU transformer
backend when no GPU backend is compiled or usable. Explicit backend names are
for benchmarking and diagnostics.

CUDA runtime policy flows through the shared `NN::BackendConfig` seam and UCI
options (`NNCudaDevice`, `NNCudaGraphExecution`, `NNCudaStableExecutionBatchSize`,
`NNCudaDeterministicAttentionSoftmax`, `NNCudaFullBufferClear`). Metal and
portable CPU ignore these CUDA-specific fields; MCTS/Hybrid cache keys include
them so a runtime-policy change cannot reuse a stale loaded network. With
`MCTSMinibatchSize=0`, CUDA search batches track the graph-replay stable batch
size (`MCTSCudaAutoMinibatchSize` → `NNCudaStableExecutionBatchSize` →
`METALFISH_CUDA_STABLE_EXECUTION_BATCH_SIZE` → 16).

## Cloud Test Matrix

The goal is to match common chess-engine tournament environments while keeping
cloud spend controlled.

| Matrix leg | Suggested runner | Purpose |
| --- | --- | --- |
| macOS arm64 Metal | GitHub-hosted macOS | Release path, Apple parity, and Metal NN artifact |
| Linux x86-64 CPU | GitHub Ubuntu or GCP C3/N2 | AB build/test and portable UCI |
| Linux x86-64 CUDA | GCP `g2-standard-8` + NVIDIA L4 | CUDA backend correctness and NPS |
| Windows x86-64 CPU | GitHub Windows MinGW + MSVC | Portable build/UCI package and MSVC host-toolchain coverage |
| Windows x86-64 CUDA compile | GitHub Windows MSVC + CUDA Toolkit | NVCC/MSVC compile coverage and release-candidate CUDA package smoke |
| Windows x86-64 CUDA runtime | Ephemeral Windows G2/L4 vWS VM | Packaged CUDA/Hybrid smoke before release |

Cloud GPU runners are created only for a benchmark run, write logs and artifacts
to Cloud Storage, then stop/delete themselves. The CUDA GPU gates are manual on
purpose: every pull request still gets Linux/Windows portable coverage, while
real NVIDIA runtime validation is started when CUDA code or backend contracts
change.

### Parity principle

Every CUDA runtime gate hard-compares against the **same-commit** macOS Metal
artifacts (BT4 and legacy) before spending GPU time: a CUDA gate will not start
its L4 VM until `tools/fetch_*_runtime_inputs.py` has fetched a successful
same-commit `MetalFish CI` (Metal) run. Numeric parity is asserted against the
Metal reference (value / WDL / moves-left / policy), and the stable-batch
eval-time ratio is bounded by `METALFISH_MAX_CUDA_METAL_EVAL_MS_RATIO`
(default `1.0`). Per-run numeric deltas and benchmark timings live in the CI
logs and Cloud Storage artifacts for each run, not in this document.

### GCP scaffolding

| Resource | Value |
| --- | --- |
| Project | `metalfish` |
| Region | `us-central1` |
| Artifact Registry | `us-central1-docker.pkg.dev/metalfish/metalfish` |
| Build artifact bucket | `gs://metalfish-build-artifacts-<project-number>` |
| Runner service account | `<runner-sa>@metalfish.iam.gserviceaccount.com` |
| Enabled APIs | Compute Engine, Cloud Build, Cloud Run, Artifact Registry, Secret Manager |

| GitHub secret/variable | Kind | Purpose |
| --- | --- | --- |
| `GCP_CREDENTIALS_JSON` | Secret | Service-account JSON for `google-github-actions/auth` |
| `METALFISH_GCP_PROJECT` | Repository variable | Optional project override; defaults to `metalfish` |

### Gates

| Gate | Workflow / config | Trigger |
| --- | --- | --- |
| macOS Metal | `.github/workflows/ci.yml` | Every PR; produces the Metal NN parity + benchmark artifacts other gates compare against |
| Portable Linux/Windows CPU | `.github/workflows/portable-ci.yml` | Every PR |
| Hybrid regression | `.github/workflows/hybrid-regression.yml` | Every PR (bounded) |
| Lichess puzzle regression | `.github/workflows/lichess-puzzles.yml` | Every PR |
| Linux CPU cloud build | `cloudbuild/linux-cpu.yaml` | Cloud Build |
| CUDA entrypoint compile/test | `cloudbuild/cuda-entrypoint.yaml` | Cloud Build |
| Linux CUDA GPU runtime | `.github/workflows/cuda-gpu-gate.yml` | Manual; requires a same-commit Metal CI run |
| Windows CUDA compile | `.github/workflows/windows-cuda-compile.yml` | Produces the self-smoked Windows MSVC+CUDA package |
| Windows CUDA runtime | `.github/workflows/windows-cuda-runtime-gate.yml` | Manual; requires same-commit Windows compile + Metal CI runs |
| CUDA release promotion | `.github/workflows/cuda-release.yml` | Manual; attaches validated Linux/Windows CUDA packages |

### Running the CUDA gates

Dispatch the manual Linux and Windows CUDA runtime gates from the same branch
tip without hand-copying run IDs:

```bash
python3 tools/dispatch_cuda_runtime_gates.py --target both --ref cuda-support
```

The dispatcher resolves the successful same-commit `MetalFish CI` run and, for
Windows, the `Windows CUDA Compile Gate` run before calling `gh workflow run`.
Use `--dry-run` to print resolved inputs without starting cloud VMs. GitHub can
dispatch only workflows already on the default branch, so validate newly added
workflow files with the direct runner instead:

```bash
python3 tools/run_cuda_runtime_gates_direct.py --target both --ref cuda-support
```

The direct runner creates a clean detached worktree at the target SHA, resolves
the same-commit Metal and Windows compile artifacts, downloads the weights the
Windows package smoke needs, then calls `tools/run_gcp_cuda_gpu_gate.sh` and
`tools/run_gcp_windows_cuda_runtime_gate.sh` with artifact collection enabled
(default root `results/cuda_runtime_direct/<sha>/`). Use
`tools/audit_cuda_gcp_resources.py --project metalfish` to list matching gate
VMs, adding `--older-than-hours <hours> --delete` to remove stale instances.

### Release promotion

Once both CUDA runtime gates have passed for the same SHA, promote the Linux and
Windows CUDA packages:

```bash
python3 tools/dispatch_cuda_release_artifacts.py \
  --ref main \
  --gate-ref main \
  --expected-sha <commit-sha> \
  --tag-name v1.0.0 \
  --attach-to-release
```

For branch-local direct GCP validation, promote the same artifacts from a direct
runtime root instead of GitHub run IDs:

```bash
python3 tools/dispatch_cuda_release_artifacts.py \
  --direct-runtime-root results/cuda_runtime_direct/<sha> \
  --tag-name v1.0.0 \
  --out-dir results/cuda_release_artifacts/<sha>
```

`tools/fetch_cuda_release_artifacts.py` rejects failed runs, wrong workflow
types, SHA mismatches, missing CUDA packages, and manifest drift, and uses
`tools/check_cuda_runtime_manifest.py` to reject runtime manifests whose
remote/runtime, BT4, legacy, benchmark, or final comparison status is not `0`.

## Release Promotion Checklist

1. Keep CUDA BK.07 tactical bestmove smokes matching the macOS Metal CI
   sentinel green on both Linux and Windows runtime gates.
2. Keep CUDA Hybrid clock-safety smokes matching the Metal CI low-clock
   boundary green: one search that starts Hybrid and one search that
   intentionally falls back before transformer time becomes unsafe.
3. Promote same-commit Linux CUDA and Windows CUDA packages through the
   release-facing audit path with manifest, hash, runtime-policy, Metal
   comparison, and Hybrid clock-evidence validation.
4. Keep `directml` explicitly deferred until CUDA Linux/Windows gates remain
   stable on package, numeric, tactical, and Hybrid smoke coverage.
5. Before calling CUDA strength-ready, require green portable CPU CI, macOS
   Metal CI, Linux L4 CUDA runtime, Windows CUDA compile, Windows L4 CUDA
   runtime, and the bounded Hybrid regression gate on the same branch tip.
