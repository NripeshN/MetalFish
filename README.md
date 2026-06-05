# MetalFish

MetalFish is a UCI chess engine built around a hybrid search: CPU alpha-beta
with NNUE plus GPU MCTS with an Lc0-compatible transformer. The mature path is
Apple Silicon with Metal/MPSGraph, and the CUDA branch brings the same shared
NN/search contracts to NVIDIA Linux and Windows hosts.

Current platform surface:

| Platform | Backend | Status |
| --- | --- | --- |
| macOS Apple Silicon | Metal/MPSGraph + CPU NNUE | Primary v0.1.0-alpha release path |
| Linux x86-64 NVIDIA | CUDA + CPU NNUE | Parity-gated in CI/GCP runtime gates |
| Windows x86-64 NVIDIA | CUDA + CPU NNUE | MSVC/NVCC compile gate plus L4 runtime gate |
| Portable Linux/Windows CPU | CPU AB and transformer fallback | Correctness/fallback, not a strength target |

## Project Status

MetalFish is experimental engine research, not a Stockfish replacement claim.
The standalone alpha-beta search is Stockfish-family search engineering; the
standalone MCTS path is an Lc0-compatible transformer search implementation.
The main research contribution is the hybrid runtime: CPU NNUE alpha-beta and
GPU transformer MCTS search the same position concurrently, exchange root-level
signals, and use a coordinator to select the final move.

## Engine Modes

| Mode | UCI option | Purpose |
| --- | --- | --- |
| Hybrid | `UseHybridSearch true` | Primary engine for play |
| Alpha-Beta | `UseHybridSearch false`, `UseMCTS false` | CPU fallback and diagnostics |
| MCTS | `UseMCTS true` | Transformer search diagnostics |

Hybrid runs AB and MCTS in parallel. AB uses CPU NNUE and the transposition
table; MCTS uses the BT4 transformer through `NNBackend`: Metal/MPSGraph on
Apple Silicon, CUDA on supported NVIDIA Linux/Windows hosts, and portable CPU
only as a fallback/correctness path. The two searches share root signals and
the coordinator chooses the final move from both engines.

## Benchmarking Policy

Benchmark results in this repository must use a fair, strength-first resource
policy:

- One shared worker-thread budget per engine.
- On Apple Silicon, `auto` uses the performance-core count.
- CPU engines receive the same `Threads` and `Hash` settings where supported.
- Pure MetalFish MCTS keeps its engine-recommended Apple worker cap unless a
  test is explicitly measuring throughput scaling. For the current BT4/MPSGraph
  backend this is one search worker with direct evals; `MCTSParallelSearch`
  must be enabled explicitly for pure-MCTS throughput experiments.
- Lc0 receives the requested `Threads` value for its own backend.
- MetalFish Hybrid receives the same total worker budget split between MCTS and
  AB workers.
- Tactical-suite scores are reported as tactical accuracy only, not Elo.

Run the fair Bratko-Kopec tactical sweep on the current machine:

```bash
python3 tests/paper_benchmarks.py --tactical --movetime 5000 \
  --threads auto --hash auto \
  --engines metalfish-ab,metalfish-mcts,metalfish-hybrid,lc0,stockfish
```

The script writes machine-readable results to `results/paper_tactical.json` and
a report to `results/paper_summary.md`. Any headline strength claims should be
based on those generated artifacts or on a larger cutechess tournament, not on
stale README tables.

For noisy tactical changes, add repeats and compare aggregate run counts:

```bash
python3 tests/paper_benchmarks.py --tactical --movetime 5000 \
  --threads auto --hash auto --tactical-repeat 3
```

For broad tactical regression testing, use an offline sample from the public
Lichess puzzle CSV instead of the 24-position smoke suite. This avoids API
quota and lets CI compare the same positions on `main` and the PR branch:

```bash
curl -L https://database.lichess.org/lichess_db_puzzle.csv.zst \
  | zstd -dc \
  | python3 tools/filter_lichess_puzzle_csv.py \
      --out results/lichess_puzzles/sample.csv \
      --max-puzzles 1000 --min-puzzles 1000 \
      --min-rating 1200 --max-rating 2800 --min-popularity 70 \
      --themes "advantage,mate,fork,sacrifice,skewer,pin,discoveredAttack"

python3 tools/lichess_puzzle_runner.py \
  --offline-csv results/lichess_puzzles/sample.csv \
  --engine build/metalfish \
  --mode hybrid \
  --weights networks/BT4-1024x15x32h-swa-6147500.pb \
  --threads auto --hash-mb 4096 --movetime-ms 1000
```

Latest local tactical run, M2 Max, 5 seconds per position, generated
2026-05-20 with `--threads auto --hash auto`:

| Engine | Score | Completed | Notes |
| --- | ---: | ---: | --- |
| MetalFish Hybrid | 24/24 | 24/24 | 1 MCTS worker + 7 AB workers, BT4 weights |
| Stockfish reference | 21/24 | 24/24 | 8 workers, 4096 MB hash |
| MetalFish AB | 20/24 | 24/24 | 8 workers, 4096 MB hash |
| MetalFish MCTS | 20/24 | 24/24 | BT4 weights, one Apple MCTS worker, strength KLD profile |
| Lc0 with BT4 weights | 17/24 | 24/24 | Metal backend, `Threads=8`, `Temperature=0` |

In this run, Hybrid solved BK.07, BK.09, BK.11, BK.17, and BK.22 where at
least one standalone component or reference engine missed. This is a tactical
suite result only; it is not an Elo estimate.

A forced full-worker pure-MCTS stress run (`MCTSMaxThreads=8`) is not a
strength profile for the current backend. It previously scored 12/24 and timed
out on BK.24, which is why pure MCTS now caps Apple workers unless
`MCTSParallelSearch=true` is set.

MetalFish's pure-MCTS strength profile intentionally uses a small
`MCTSMinimumKLDGainPerNode=0.00005` tactical stopper and
`PureMCTSSmartPruningFactor=1.33`. For an exact Lc0-style KLD-off diagnostic
comparison, run:

```bash
python3 tests/bk_parity.py --engine both --movetime 5000 \
  --threads 8 --mcts-kld 0.0
```

This is a tactical-suite result only. It is useful for regression tracking, but
it is not an Elo estimate.

For game-strength testing, prefer cutechess matches with the same resource
policy:

```bash
THREADS=8 HASH=4096 ./tools/run_cutechess_tournament.sh \
  --games=20 --tc=300+0.1 --concurrency=1
```

## Platform Acceleration

- Shared NN execution-plan, tensor-layout, output-decoding, and policy-map
  contracts across Metal and CUDA
- Metal/MPSGraph transformer inference
- CUDA transformer inference on NVIDIA Linux/Windows hosts
- Unified-memory CPU/GPU search architecture
- Accelerate/vDSP policy softmax
- NEON and ARM dot-product NNUE kernels
- 128-byte aligned MCTS nodes for Apple Silicon cache lines
- Native Apple CPU tuning and Release LTO by default
- Local Polyglot opening books for the Lichess bot
- Syzygy 3-4-5 tablebase support
- Experimental Core ML/ANE root-probe sidecar, disabled by default

## Requirements

Common:

- CMake 3.20 or newer
- C++20 compiler
- Python 3.11+ for bot and benchmark tooling

macOS Metal:

- Apple Silicon Mac, arm64
- macOS 13 or newer
- Xcode Command Line Tools
- Homebrew packages: `protobuf zlib abseil`

```bash
brew install cmake protobuf zlib abseil
python3 -m pip install -r tests/requirements.txt
```

Linux CUDA:

- NVIDIA GPU with a supported CUDA toolkit
- CUDA Toolkit 12.x, cuBLAS, protobuf, zlib, and Abseil development packages

Windows CUDA:

- Windows x86-64 with Visual Studio 2022/MSVC
- CUDA Toolkit 12.x
- vcpkg or the repository CI scripts for protobuf/zlib/Abseil dependencies

## Network Files

MetalFish expects network files under `networks/`.

```bash
mkdir -p networks
python3 tools/download_engine_networks.py --dest networks
```

Manual download equivalent:

```bash
curl -L -o networks/nn-c288c895ea92.nnue \
  https://tests.stockfishchess.org/api/nn/nn-c288c895ea92.nnue
curl -L -o networks/nn-37f18f62d772.nnue \
  https://tests.stockfishchess.org/api/nn/nn-37f18f62d772.nnue
curl -L -o networks/BT4-1024x15x32h-swa-6147500.pb.gz \
  https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1024x15x32h-swa-6147500.pb.gz
gzip -dc networks/BT4-1024x15x32h-swa-6147500.pb.gz \
  > networks/BT4-1024x15x32h-swa-6147500.pb
```

## Build

macOS Metal:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_METAL=ON
cmake --build build --target metalfish -j"$(sysctl -n hw.ncpu)"
```

Linux CUDA:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON
cmake --build build --target metalfish --parallel
```

Windows CUDA builds are validated through the MSVC/NVCC gate scripts and
GitHub workflow. For release-candidate packages, use the CUDA compile/runtime
gates documented in `docs/cross_platform_backend_plan.md`.

## Run Hybrid

```bash
./build/metalfish
```

Recommended UCI setup:

```text
setoption name UseHybridSearch value true
setoption name NNWeights value networks/BT4-1024x15x32h-swa-6147500.pb
setoption name Threads value 8
setoption name Hash value 4096
setoption name Ponder value true
setoption name HybridABCandidateVerifyMs value 120
isready
position startpos
go movetime 5000
```

## Experimental ANE Root Probe

The Core ML/ANE path is an experimental Hybrid sidecar. It does not replace the
BT4 transformer MCTS and it is not enabled by default. The retained profile uses
the smaller T1-512 Lc0 network on `cpu-ne` as root evidence while BT4 continues
to run on Metal/MPSGraph:

```text
setoption name HybridANERootProbe value true
setoption name HybridANERootHints value false
setoption name HybridANEWeights value networks/t1-512x15x8h-distilled-swa-3395000.pb.gz
setoption name HybridANEModelPath value build/coreml/compiled/t1-512-heads-b8.mlmodelc
setoption name HybridANEComputeUnits value cpu-ne
setoption name HybridANEOnlyPawnEndgames value false
setoption name HybridANERootHintCount value 10
setoption name HybridANERootHintWaitMs value 0
setoption name HybridANEMinBudgetMs value 0
```

Current ANE findings:

- `cpu-ne` is the only retained Core ML compute-unit profile. `all` and
  `cpu-gpu` are faster in isolation but compete with Metal inference.
- T1-512 is retained over T1-256. T1-256 is lower latency, but it regressed the
  ANE-sensitive repeat gate.
- The Lichess bot now probes all roots when ANE is explicitly enabled. The
  broader scope tied the retained 200-puzzle gate at 199/200 and improved the
  current ANE-sensitive low-root repeat gate from 30/40 to 34/40. UCI still
  defaults to the conservative pawn-only scope unless the option above is set.
- ANE root hints are available as an explicit experiment, but they are disabled
  by default. After the low-visit pawn-lever update, ANE root hints regressed
  the local 1s BK repeat from 67/72 to 66/72, while ANE probe without root
  hints matched the retained non-ANE profile. The retained wait profile remains
  `250 ms` for explicit root-hint runs.
- ANE is retained as a confirming sidecar, not as a third equal engine. Puzzle
  summaries report `ANE searches`, non-empty roots, top moves, failures,
  agreement with MCTS, and ANE-confirmed MCTS overrides so runs distinguish
  “configured” from “actually contributed.”

Useful local probes:

```bash
python3 tools/lichess_puzzle_runner.py \
  --offline-csv results/judge_ane_broad_1000ms/hard200_sample.csv \
  --engine build/metalfish --mode hybrid \
  --weights networks/BT4-1024x15x32h-swa-6147500.pb \
  --hybrid-ane-root-probe \
  --hybrid-ane-weights networks/t1-512x15x8h-distilled-swa-3395000.pb.gz \
  --hybrid-ane-model-path build/coreml/compiled/t1-512-heads-b8.mlmodelc \
  --hybrid-ane-compute-units cpu-ne \
  --threads 8 --hash-mb 4096 --movetime-ms 3000

python3 tools/lc0_coreml_concurrency_benchmark.py \
  networks/t1-512x15x8h-distilled-swa-3395000.pb.gz \
  --metal-probe build/metalfish_nn_probe \
  --metal-weights networks/BT4-1024x15x32h-swa-6147500.pb \
  --coreml-compute-unit cpu-ne --batch-size 8

python3 tools/compare_puzzle_runs.py \
  --baseline results/baseline.jsonl \
  --candidate results/candidate.jsonl \
  --match-repeat-ids --ane-summary
```

## Lichess Bot

The bot is configured around the Hybrid engine, local opening books, Syzygy,
pondering, crash recovery, and resource-aware thread/hash sizing.

Download the included local Polyglot book set:

```bash
python3 tools/opening_book_manager.py download --book all
```

Run a rated seek with an Elo floor:

```bash
python3 tools/lichess_bot.py --seek --rotate --no-casual \
  --accept-rated --elo-seek --min-rated-opponent-elo 2200
```

The bot will load `books/*.bin` automatically. Online Explorer fallback is off
by default; enable it only when needed with `METALFISH_BOOK_ALLOW_ONLINE=true`.

## Tests

macOS:

```bash
cmake --build build --target metalfish_tests -j"$(sysctl -n hw.ncpu)"
./build/metalfish_tests
python3 tests/testing.py --quick
python3 tests/test_lichess_bot.py
python3 tests/test_lichess_puzzle_runner.py
python3 tests/test_ponder_stress.py --smoke
python3 tests/paper_benchmarks.py --tactical --movetime 5000 \
  --threads auto --hash auto
```

Portable Linux/Windows CPU and CUDA builds are covered by the GitHub workflows
and GCP runtime gates. CUDA strength checks should use the same-commit Metal,
Linux CUDA, and Windows CUDA artifacts so backend parity is measured against
the same weights, search options, and source SHA.

## Strength Claims

Use precise language when discussing results:

- Valid: "MetalFish Hybrid solved X/24 Bratko-Kopec positions at 5s/position on
  an M2 Max using the fair benchmark script."
- Valid: "MetalFish MCTS loads the same BT4 Lc0 weights and uses compatible
  input/policy mapping, but has its own Metal/CUDA runtimes and search
  controls."
- Valid: "The hybrid engine combines CPU alpha-beta and GPU MCTS with live
  root-signal sharing and evidence-based final arbitration."
- Invalid without larger testing: "MetalFish is stronger than Stockfish" or
  "MetalFish has higher Elo than Lc0."

## Repository Layout

```text
src/core/      Board representation and move generation
src/eval/      NNUE evaluation and Apple Silicon CPU kernels
src/nn/        BT4 transformer loader, encoder, Metal/MPSGraph, CUDA backends
src/search/    Alpha-beta search
src/mcts/      Lc0-style MCTS search
src/hybrid/    Parallel Hybrid coordinator
src/uci/       UCI protocol
tests/         Unit, UCI, ponder, and benchmark tests
tools/         Lichess bot, puzzle runner, books, Syzygy, tournaments
```

## License

MetalFish is released under GPL-3.0. See [LICENSE](LICENSE).
