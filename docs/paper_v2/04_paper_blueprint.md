# 04 — Paper blueprint (Chapter Plan + INSIGHT collection)

Produced in ARS `plan` mode. The Socratic questions of the protocol were answered from
the evidence in `01_…` and `02_…`; each answer the author must confirm is marked
**[author to confirm]**.

## 1. Publication situation and integrity actions

- The IntelliSys 2026 chapter (DOI 10.1007/978-3-032-32726-0_39) is published. A rewrite
  cannot be "resubmitted" to the same venue; it must be a **new paper** that (a) cites the
  chapter, (b) states explicitly what is new and what it corrects, and (c) contains well
  over the customary ≥30% new material (here essentially all results are new).
- **Correction of the record [author decision]**: the chapter contains factual errors that
  are not matters of interpretation (Elo figures from an invalid method; 64-byte nodes;
  classifier weighting; cache/TT sizes; NNUE sizes). Springer Nature's policy allows an
  author-initiated Correction via a support ticket with an annotated PDF. Recommended:
  file a Correction for the factual errors once the new measurements exist, and cite the
  Correction from the new paper. This is outward-facing and is the author's call; the plan
  does not depend on it, but the new paper's §8 is written either way.
- **Double-anonymous venues** (IEEE Transactions on Games since 2025): refer to the chapter
  in the third person ("Niketan [n] reported…"), anonymise repository and Zenodo links at
  submission (anonymous GitHub / Zenodo restricted link), and keep the correction
  statement neutral.

## 2. Thesis (outcome-neutral) and pre-written framings

**Research statement.** Running an Lc0-class transformer MCTS on the GPU concurrently
with a Stockfish-class alpha-beta search on the CPU of a consumer unified-memory SoC,
with a final arbiter, changes playing strength by Δ Elo (95% CI) relative to the
alpha-beta search alone under equal wall-clock and equal CPU-thread budgets; the sign
and size depend on CPU budget and GPU throughput in a measurable way, and the mechanism
is explained by the thin-tree move quality of batch-1 MCTS and by the arbiter's
disagreement behaviour.

Pre-written framings for RQ1a (choose one after unblinding):

- **Positive** (lower CI bound > 0): "A GPU search that is individually ~hundreds of Elo
  weaker than the CPU search still adds Δ Elo when arbitrated against it at equal
  budgets; the gain concentrates in [regime from RQ3/RQ4]."
- **Null** (CI spans 0): "At the resolution of ±X Elo, the GPU search adds nothing
  measurable to a Stockfish-class CPU search on this hardware; the oracle analysis shows
  why: on the N% of moves where the two disagree, the MCTS move is better in only p% of
  cases, bounding any arbiter's gain to ≤ Y Elo."
- **Negative** (upper CI bound < 0): "The hybrid costs Δ Elo: the two CPU threads
  surrendered to the GPU path are worth more than the GPU path returns (RQ2 decomposition)."

## 3. Contributions (as stated in the paper)

1. A **controlled, pre-registered evaluation methodology** for hybrid/ensemble chess
   engines sharing one machine: fair-resource policy, pentanomial/nElo statistics, UHO
   openings, GPU-active telemetry, component isolation (RQ2) and scaling curves (RQ3/RQ4),
   with all artifacts released.
2. The **as-built MetalFish system** on Apple Silicon: a concurrent AB+MCTS runtime with
   lock-light shared root state, MCTS→AB root-order hints, and a gated arbiter; the BT4
   transformer on MPSGraph; measured batch-1 latency ceiling and CPU–GPU contention.
3. **Measurements** answering RQ1–RQ8, including the marginal-value result, the
   CPU-budget crossover, the MCTS-throughput projection, component parity with Lc0 and
   Stockfish, and suite accuracies with leakage caveats.
4. An **oracle analysis of arbitration** quantifying the upper bound of any arbiter on
   this component pair and the realised arbiter's accuracy.
5. A **correction** of the earlier chapter's claims, with the root causes (§8).

## 4. Title candidates [author to confirm]

1. "Does a GPU Help a Stockfish-Class Engine? A Controlled Evaluation of Concurrent
   Alpha-Beta and Transformer-MCTS Search on Apple Silicon"
2. "The Marginal Value of GPU Monte-Carlo Tree Search alongside Alpha-Beta on Unified
   Memory: Methodology, Measurements and an Oracle Analysis"
3. "When Two Searches Disagree: Measuring Hybrid Alpha-Beta/MCTS Chess Search on
   Consumer Unified-Memory Hardware"

## 5. Target venue [author to confirm]

| Venue | Fit | Constraints | Timing |
| --- | --- | --- | --- |
| **IEEE Transactions on Games** (primary) | Journal; "AI for games" in scope; extended versions of conference papers accepted when the increment is stated | Double-anonymous since Jan 2025; article length flexible (≈10–14 pages two-column) | Rolling; target submission mid-December 2026 |
| **ICGA Journal** (alternative) | The computer-chess community's journal; single-blind; strong fit for engine methodology | Lower impact factor; quarterly | Rolling |
| **IEEE Conference on Games 2027** (conference fallback) | AI-for-game-playing track | Full papers due 2027-03-01 (8 pages) | Aizuwakamatsu, Aug 2–5, 2027 |
| Advances in Computer Games 2027 | ICGA conference, odd years, LNCS | ~10 pages; deadline expected Aug–Sep 2027 (unconfirmed) | Later |

Post an arXiv preprint after submission if the venue allows (ToG does).

## 6. Chapter plan (journal version, ≈9,500 words + figures)

| § | Title | Words | Core argument | Evidence |
| --- | --- | --- | --- | --- |
| 1 | Introduction | 900 | Two paradigms; unified memory makes concurrent AB+MCTS cheap to try; prior claims (incl. ours) lacked controlled measurement; state RQs and contributions | Audit §A; protocol §5 |
| 2 | Background and related work | 1,400 | AB+NNUE; PUCT MCTS and BT4; MCTS–minimax hybrids (Baier & Winands; Lanctot; Huang's rollout view); engines combining both (Scorpio, Komodo MCTS, SF MCTS branch, lc0-stockfish-hybrid); MoE/ensemble in MCTS (Czech et al.); why MCTS struggles with traps (Ramanujan et al.); evaluation methodology (pentanomial, nElo, UHO, Ordo, hardware normalisation) | `05_related_work.md` |
| 3 | System as built | 1,800 | Fig. 1 architecture; AB = Stockfish port (deltas listed); MCTS engine and NN backend (MPSGraph, FP16, batch-1); hybrid runtime (threads, shared state, hints, verification, arbiter, time-safety rule); what is *not* active by default | 03 §0; audit §C |
| 4 | Evaluation methodology | 1,800 | Fair-resource policy; configurations; fastchess protocol; statistics; RQs and designs; controls; pre-registration | protocol §§2–7 |
| 5 | Results | 2,500 | 5.1 Microbenchmarks (M1–M6); 5.2 RQ1 primary + secondary TCs + balanced-book control; 5.3 RQ2 decomposition; 5.4 RQ3 CPU crossover; 5.5 RQ4 throughput scaling; 5.6 RQ5 component checks; 5.7 RQ7 suites; 5.8 RQ8 relative gauntlet; 5.9 RQ9 deployment | `results/paper_v2/` |
| 6 | Why: arbitration and oracle analysis | 1,200 | Disagreement rate, visits at decision, reason histogram; oracle outcomes; upper bound on any arbiter; Hybrid-min ablation; implications for hybrid design | RQ6 |
| 7 | Threats to validity | 500 | Single platform; book; GPU-active fraction; leakage; reference drift; adjudication | protocol §7 |
| 8 | Relation to the IntelliSys 2026 chapter | 300 | Table of corrected claims and root causes | audit §F |
| 9 | Conclusion and future work | 400 | What was learned; conditions under which a hybrid is worth it; CUDA/other GPUs; learned arbiters | — |
| — | Data/code availability, AI-use declaration, acknowledgements (Stockfish, Lc0 upstreams; GPL) | 200 | | |

Conference variant (8 pages): drop §5.7–5.9 to an appendix/supplement, compress §2 and §3.

## 7. Figures and tables

Figures: F1 as-built architecture (threads, data flows, what is default-off); F2 BT4
eval latency/throughput vs batch size, FP16 vs FP32, MetalFish vs Lc0 (M1); F3 AB NPS
and time-to-depth vs threads (M2) and in-game MCTS evals/s vs AB threads (M3); F4
GPU-active fraction and MCTS playouts/move vs TC (M5); F5 Elo difference Hybrid−AB vs
CPU threads with CIs (RQ3); F6 Elo difference vs MCTS throughput (RQ4); F7 decision-reason
histogram and oracle outcomes on disagreements (RQ6); F8 Ordo relative ratings with error
bars (RQ8).

Tables: T1 configurations and fair-resource policy; T2 primary results per cell
(pairs, pentanomial counts, draw %, Elo ± CI, nElo, LOS, time losses, adjudications,
GPU-active %); T3 component checks (RQ5); T4 suites with Wilson CIs and leakage caveat;
T5 corrected claims vs the earlier chapter; T6 compute spent per cell (hours, games).

## 8. INSIGHT collection (plan-mode Socratic answers; author to confirm each)

- **thesis_statement**: see §2. [author to confirm]
- **research_gap**: no controlled measurement exists of the *marginal* value of a GPU
  neural MCTS added to a top-class CPU alpha-beta engine under equal hardware budgets;
  prior hybrid work embeds minimax in MCTS or vice versa and evaluates on small games or
  with uncontrolled Elo.
- **urgency**: unified-memory SoCs (Apple M-series, AMD Strix Halo, NVIDIA Grace-class)
  make concurrent CPU+GPU search a default-available design; without measurement, the
  community (and our own earlier paper) over-attributes strength to the GPU path.
- **what the reader should think differently about**: "hybrid" strength claims must be
  decomposed into component strength, resource cost, and arbiter quality; a GPU search can
  be hundreds of Elo weaker and still be *useful* only if the arbiter can identify the
  rare positions where it is right — and the oracle bound says how much that is worth.
- **strongest counter-argument**: "a stronger MCTS (bigger GPU, better batching) would
  change the conclusion" → addressed by RQ4's throughput curve and RQ5a's parity with Lc0;
  "the arbiter is bad, not the idea" → addressed by RQ6's oracle upper bound and the
  Hybrid-min ablation.
- **biggest methodological limitation**: one hardware platform; mitigated by scaling
  curves, not removed.
- **the one thing to remember**: the question "does the GPU help?" has a measurable
  answer with error bars, and the answer depends on where the CPU budget ends and what
  the disagreement oracle says.
- **unexpected result handling**: if RQ3 shows a crossover at T≤4, the paper's practical
  recommendation changes to "enable the hybrid on low-core parts only" — pre-registered.
- **contribution_claim** (in the author's words): [author to write after reading §3]

## 9. Writing process

1. After MUST cells complete: run ARS `outline-only` on this blueprint + `results/paper_v2`
   summaries to produce the evidence map; then `full` mode for the draft (LaTeX, IEEE
   two-column template for ToG), with the Writing Quality Check and acronym check.
2. Internal review with `academic-paper-reviewer` (methodology-focus mode, then full panel);
   address Critical/Major items; second pass.
3. `citation-check` on the final bibliography (every DOI/arXiv ID verified; the
   to-verify items in `05_related_work.md` resolved or dropped).
4. `disclosure` mode for the venue's AI-use statement (the chapter already declares
   generative-AI assistance for code and drafting; keep that honesty).
5. Reproducibility package: Zenodo deposit (PGNs, JSON, scripts, hashes), repository tag,
   README with one-command regeneration of every figure.

## 10. Timeline (assuming start 2026-10-08)

| Week | Dates | Work |
| --- | --- | --- |
| 1 | Oct 8–14 | P0 fixes (C1–C12), tooling (C7–C11a), unit tests; pilot; protocol freeze tag |
| 2–3 | Oct 15–28 | MUST cells (≈185 h machine time) interleaved; oracle study on completed games; microbenchmarks |
| 4–5 | Oct 29–Nov 11 | SHOULD cells (≈370 h; halve N or add a second machine if needed); RQ7 suites; Berserk/Patricia rebuilds for RQ8 |
| 6 | Nov 12–18 | Analysis, figures, tables; decide framing (§2); write §§3–6 |
| 7–8 | Nov 19–Dec 2 | Full draft; internal reviewer panel; revisions; citation check |
| 9 | Dec 3–12 | Final checks, Zenodo deposit, anonymisation; submit to IEEE ToG (or hold for CoG 2027 by Mar 1) |

Dependencies: C11 (ablation switch) gates RQ6c only; C12 gates RQ4 only; everything else
can start after the freeze. If the machine is also needed for development, run cells
overnight and on weekends and extend weeks 2–5 accordingly.
