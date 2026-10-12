# Paper v2: rewrite and re-evaluation plan

This directory is the planning package for rewriting the MetalFish paper with a
proper, pre-registered evaluation. It was produced on 2026-10-07 from a full read of
the published paper (branch `paper`, commit `85d0a01`), a line-level audit of the
engine as built (branch `post-v1-tuning`, `v1.0.0-65-gb935e66`), the artifacts under
`results/`, a literature search, and a verified end-to-end run of the new evaluation
toolchain (fastchess + Ordo + Stockfish 19 + Lc0 0.32.1 + UHO books + test suites).

## The published paper

| Field | Value |
| --- | --- |
| Title | Hybrid Chess Search: Combining Alpha-Beta and Monte Carlo Tree Search on Unified Memory Architectures |
| Author | Nripesh Niketan (ORCID 0009-0008-2066-1937) |
| Venue | Intelligent Systems and Applications (IntelliSys 2026, Vol. 2), Lecture Notes in Networks and Systems, Springer Nature Switzerland, Cham |
| DOI | 10.1007/978-3-032-32726-0_39 |
| Pages | 653–664, published online 2026-08-14 (print year 2027) |
| ISBN | 978-3-032-32725-3 (print), 978-3-032-32726-0 (online) |
| References | 16 |
| Source | `git show paper:metalfish/metalfish.tex` (LaTeX, `svproc.cls`), PDF draft at `paper:metalfish/PDF/metalfish.pdf` |

## Documents

| File | Purpose |
| --- | --- |
| [01_published_paper_audit.md](01_published_paper_audit.md) | Claim-by-claim audit of the published paper against code, artifacts, and fresh measurements. Verdict per claim. |
| [02_evaluation_protocol.md](02_evaluation_protocol.md) | Pre-registered evaluation protocol: research questions, hypotheses, designs, sample sizes, statistics, controls, decision rules, compute budget. |
| [03_codebase_changes.md](03_codebase_changes.md) | Engineering plan: fixes and tooling required before and during the experiments, with priorities, file pointers, and acceptance criteria. |
| [04_paper_blueprint.md](04_paper_blueprint.md) | Thesis, contributions, chapter plan with word budgets, figure/table list, venue analysis, integrity actions (correction of the LNNS chapter), timeline. |
| [05_related_work.md](05_related_work.md) | Literature map with identifiers (DOI/arXiv) and what each source is used for; items still to verify are marked. |
| [06_environment_and_baselines.md](06_environment_and_baselines.md) | Hardware/software environment, installed tooling, and the baseline measurements taken on 2026-10-07 (NN latency sweeps, Lc0 batch sweep, fastchess smoke test). |
| [experiments.json](experiments.json) | Machine-readable manifest of every experiment cell (engines, options, TC, book, pairs, priority, estimated hours) for the runner to consume. |

Third-party tooling lives under the gitignored `reference/` directory and is
documented in `reference/TOOLING.md`.

## One-paragraph summary of the situation

The published paper's headline numbers are not supportable. The "~4048 Elo" for
MetalFish-AB and "~3724 Elo" for the hybrid come from averaging arbitrary anchor
ratings (CCRL numbers for other hardware, invented ratings for Stockfish skill levels)
over 5-game matches, with no artifacts kept. MetalFish-AB is a renamed Stockfish dev
build (2026-01-03) with the Stockfish NNUE networks, so it cannot be 250 Elo above
Stockfish; the repository's own clean match shows 22 draws in 22 games against
Stockfish. The hybrid's architecture is also described inaccurately (no decision
weighting by the position classifier, no observed agreement-based early stops,
128-byte not 64-byte nodes, different cache and TT sizes). Most importantly, every
valid measurement in the repository says the GPU transformer MCTS adds roughly zero
Elo over the alpha-beta component at the time controls tested, and the MCTS move is
selected in about 1% of hybrid moves. The new paper therefore has to be built around
a controlled, pre-registered measurement of the *marginal value* of the GPU search
under equal hardware budgets, an explanation of the result (throughput ceiling of
batch-1 inference on MPSGraph, arbitration/oracle analysis), and an honest
correction of the earlier claims. The protocol in `02_evaluation_protocol.md` is
designed so the paper is publishable whichever way the measurements come out.

## Status log

- **2026-10-12** — Scope re-set by the author: merge PR #66, make the engine as strong as
  possible, then produce an arXiv preprint from verified data. Done today: AB core ported to
  Stockfish 19 (bench parity 2,497,913; gate +62 ± 28 Elo at 10+0.1); PGO measured (no gain,
  dropped); 12-thread gate +6 ± 16 (inconclusive, 8 P-cores kept); `HybridOversubscribe`
  option added (AB keeps all threads) with its gate running; measurement tooling in
  `tools/paper/` (runner, pentanomial stats validated against fastchess, trace parser, oracle
  probe, suite runner, microbenchmarks, SPRT gate); CI fixes for the rebase (baseline nets,
  Windows include order, unsigned-enum loop, probe link stub, PR-sized tournament, path
  filters). See `07_engine_strength_program.md` for gate results.
